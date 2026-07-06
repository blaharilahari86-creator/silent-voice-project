from googletrans import Translator
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import sqlite3
import subprocess
import sys
from gtts import gTTS
import os
from functools import wraps
from uuid import uuid4
from collections import deque
import threading
import time
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from preprocess import prepare_variants, to_model_input
from sign_labels import SIGN_LABELS, SIGN_LABELS_WORDS, SIGN_LABELS_ALPHA, get_label_map

app = Flask(__name__)
PREDICTION_STATE = {}

# Load the sign language model once when the app starts.


sign_model_words = None
sign_model_alpha = None
try:
    # primary (word) model
    model_path = os.path.join(os.getcwd(), 'sign_model.h5')
    if os.path.exists(model_path):
        sign_model_words = load_model(model_path)
    # optional alphabet model
    alpha_model_path = os.path.join(os.getcwd(), 'sign_model_alpha.h5')
    if os.path.exists(alpha_model_path):
        sign_model_alpha = load_model(alpha_model_path)
    if sign_model_words is None and sign_model_alpha is None:
        print('No sign model files found at', model_path, 'or', alpha_model_path)
except Exception as e:
    print('Failed to load sign models:', e)

# Backwards compatibility: expose primary model as `sign_model` for other modules/tests
sign_model = sign_model_words
app.secret_key = 'your_secret_key_change_this'
CORS(app)

# DATABASE CONNECTION

def get_db():
    return sqlite3.connect("data.db")

def migrate_history_schema(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(history)")
    columns = [row[1] for row in cursor.fetchall()]

    if "username" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN username TEXT")
        cursor.execute("UPDATE history SET username = 'guest' WHERE username IS NULL")
        conn.commit()
    
    if "created_at" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN created_at TIMESTAMP")
        cursor.execute("UPDATE history SET created_at = datetime('now') WHERE created_at IS NULL")
        conn.commit()


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(username) REFERENCES users(username)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history_trash (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        orig_id INTEGER,
        username TEXT,
        text TEXT,
        created_at TIMESTAMP,
        deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    migrate_history_schema(conn)
    conn.close()

init_db()


# Background cleanup: remove generated audio files older than `AUDIO_TTL_SECONDS`.
AUDIO_TTL_SECONDS = 60 * 60 * 6  # 6 hours


def cleanup_audio_files(static_dir, ttl=AUDIO_TTL_SECONDS):
    now = time.time()
    try:
        for fname in os.listdir(static_dir):
            if not fname.lower().endswith('.mp3'):
                continue
            fpath = os.path.join(static_dir, fname)
            try:
                mtime = os.path.getmtime(fpath)
                if now - mtime > ttl:
                    os.remove(fpath)
            except Exception:
                pass
    except Exception:
        pass


def cleanup_audio_worker(static_dir, ttl=AUDIO_TTL_SECONDS, interval=60*30):
    while True:
        cleanup_audio_files(static_dir, ttl=ttl)
        time.sleep(interval)


# Start cleanup thread if static exists, and clean old files immediately on startup.
try:
    static_dir_path = os.path.join(os.getcwd(), 'static')
    os.makedirs(static_dir_path, exist_ok=True)
    cleanup_audio_files(static_dir_path)
    t = threading.Thread(target=cleanup_audio_worker, args=(static_dir_path,), daemon=True)
    t.start()
except Exception:
    pass


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if 'username' not in session:
            if request.method == 'POST' or request.is_json:
                return jsonify({"error": "Authentication required."}), 401
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view


# Inject `username` into all templates from the session so the sidebar
# always shows the logged-in user regardless of whether individual
# view functions pass `username` explicitly.
@app.context_processor
def inject_user():
    return dict(username=session.get('username'))


# =========================
# LOGIN PAGE
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['username'] = username
            return jsonify({"success": True, "message": "Login successful"})
        else:
            return jsonify({"success": False, "message": "Invalid credentials"})
    
    return render_template("login.html")


# =========================
# REGISTER PAGE
# =========================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return jsonify({"success": True, "message": "Registration successful"})
        except sqlite3.IntegrityError:
            return jsonify({"success": False, "message": "Username already exists"})
    
    return render_template("login.html")


# =========================
# PROFILE PAGE
# =========================

@app.route("/profile")
@login_required
def profile():
    username = session['username']
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, text, created_at FROM history WHERE username = ? ORDER BY id DESC", (username,))
    history = cursor.fetchall()
    conn.close()
    
    return render_template("profile.html", username=username, history=history)


# =========================
# LOGOUT
# =========================

@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for('login'))


# =========================
# DASHBOARD
# =========================

@app.route("/")
def dashboard():
    username = session.get('username')
    # Redirect unauthenticated users to login
    if not username:
        return redirect(url_for('login'))

    history = []
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, text, created_at FROM history WHERE username = ? ORDER BY id DESC LIMIT 5", (username,))
    history = cursor.fetchall()
    conn.close()
    return render_template("dashboard.html", username=username, history=history)


# =========================
# SPEECH TO TEXT PAGE
# =========================

# =========================
# SPEECH TO TEXT PAGE
# =========================

@app.route("/speech-to-text")
@login_required
def speech_to_text():

    return render_template(
        "speech_to_text.html"
    )


# =========================
# TEXT TO SPEECH PAGE
# =========================

@app.route("/text-to-speech")
@login_required
def text_to_speech():

    return render_template(
        "text_to_speech.html"
    )


# =========================
# SIGN LANGUAGE PAGE
# =========================

@app.route("/sign-language")
@login_required
def sign_language():
    return render_template(
        "sign_language.html",
        labels=SIGN_LABELS
    )


# =========================
# HISTORY PAGE
# =========================

@app.route("/history-page")
@login_required
def history_page():
    username = session['username']
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, username, text, created_at FROM history WHERE username = ? ORDER BY id DESC", (username,)
    )

    history = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        history=history
    )


# =========================
# SAVE DATA
# =========================

@app.route("/save", methods=["POST"])
@login_required
def save():

    data = request.get_json()

    text = data.get("text")
    username = session['username']

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO history (username, text) VALUES (?, ?)",
        (username, text)
    )

    conn.commit()

    conn.close()

    return jsonify({
        "message": "Saved successfully"
    })


@app.route('/delete-history', methods=['POST'])
@login_required
def delete_history():
    data = request.get_json() or {}
    entry_id = data.get('id')
    username = session.get('username')
    if entry_id is None:
        return jsonify({"error": "missing id"}), 400

    conn = get_db()
    cur = conn.cursor()
    try:
        # fetch the row
        cur.execute('SELECT id, username, text, created_at FROM history WHERE id = ? AND username = ?', (entry_id, username))
        row = cur.fetchone()
        if not row:
            conn.close()
            return jsonify({"deleted": False}), 200

        orig_id, uname, text, created_at = row
        # move to trash
        cur.execute('INSERT INTO history_trash (orig_id, username, text, created_at, deleted_at) VALUES (?, ?, ?, ?, datetime("now"))',
                    (orig_id, uname, text, created_at))
        trash_id = cur.lastrowid
        # delete from main table
        cur.execute('DELETE FROM history WHERE id = ? AND username = ?', (entry_id, username))
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500
    conn.close()
    return jsonify({"deleted": True, "trash_id": trash_id})


@app.route('/clear-history', methods=['POST'])
@login_required
def clear_history():
    username = session.get('username')
    conn = get_db()
    cur = conn.cursor()
    try:
        # move rows to trash
        cur.execute('INSERT INTO history_trash (orig_id, username, text, created_at, deleted_at) SELECT id, username, text, created_at, datetime("now") FROM history WHERE username = ?', (username,))
        moved = cur.rowcount
        # delete from history
        cur.execute('DELETE FROM history WHERE username = ?', (username,))
        deleted = cur.rowcount
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500
    conn.close()
    return jsonify({"cleared": True, "deleted_count": moved})


@app.route("/capture-sample", methods=["POST"])
@login_required
def capture_sample():
    data = request.get_json() or {}
    image_data = data.get("image", "")
    label = data.get("label", "").strip()

    if not image_data.startswith('data:image/'):
        return jsonify({"error": "Invalid image payload."}), 400
    if label not in SIGN_LABELS.values():
        return jsonify({"error": "Invalid label."}), 400

    try:
        encoded = image_data.split(',', 1)[1]
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Unable to decode image."}), 400

        save_dir = os.path.join(os.getcwd(), "dataset", "real_capture", label)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{label}_{int(time.time())}_{uuid4().hex[:8]}.png"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img)

        return jsonify({"message": "Sample captured.", "path": save_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_prediction_state():
    client_id = request.headers.get('X-Forwarded-For', request.remote_addr or 'local')
    client_id = client_id.split(',')[0].strip() if client_id else 'local'
    state = PREDICTION_STATE.get(client_id)
    if state is None:
        state = {
            'buffer': deque(maxlen=5),
            'stable_label': 'Detecting...',
            'stable_confidence': 0.0,
        }
        PREDICTION_STATE[client_id] = state
    return state


# =========================
# SIGN PREDICTION ENDPOINT
# =========================

@app.route("/predict-sign", methods=["POST"])
@login_required
def predict_sign():
    if sign_model_words is None and sign_model_alpha is None:
        return jsonify({"error": "No sign model loaded."}), 500

    data = request.get_json(silent=True) or {}
    image_data = data.get('image', '')
    if not isinstance(image_data, str) or not image_data.startswith('data:image/'):
        return jsonify({"error": "Invalid image payload."}), 400

    try:
        encoded = image_data.split(',', 1)[1]
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return jsonify({"error": "Unable to decode image."}), 400

        mp_hands = mp.solutions.hands
        hand_box = None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        ) as hands:
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                h_img, w_img, _ = img.shape
                for hand_landmarks in results.multi_hand_landmarks:
                    x_list = [int(lm.x * w_img) for lm in hand_landmarks.landmark]
                    y_list = [int(lm.y * h_img) for lm in hand_landmarks.landmark]
                    x_min = max(0, min(x_list) - 20)
                    y_min = max(0, min(y_list) - 20)
                    x_max = min(w_img, max(x_list) + 20)
                    y_max = min(h_img, max(y_list) + 20)
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    hand_box = (x_min, y_min, x_max, y_max)
                    break

        if hand_box is None:
            h_img, w_img = img.shape[:2]
            side = min(w_img, h_img)
            x_min = (w_img - side) // 2
            y_min = (h_img - side) // 2
            hand_box = (x_min, y_min, x_min + side, y_min + side)

        variants = prepare_variants(img, box=hand_box)

        # Helper to produce ensemble average predictions for a model.
        # We keep the variant count small and aligned with the trained data.
        def predict_ensemble(model):
            ensemble = None
            for image_variant in variants.values():
                inp = to_model_input(image_variant)
                preds = model.predict(inp, verbose=0)[0]
                if ensemble is None:
                    ensemble = np.array(preds, dtype=np.float32)
                else:
                    ensemble += preds
            if ensemble is None:
                return None
            ensemble /= len(variants)
            return ensemble

        response = {}

        # Words model prediction
        if sign_model_words is not None:
            preds_w = predict_ensemble(sign_model_words)
            if preds_w is not None:
                class_index_w = int(np.argmax(preds_w))
                conf_w = float(np.max(preds_w))
                label_w = SIGN_LABELS_WORDS.get(class_index_w, 'Unknown')
                response['words'] = {'label': label_w, 'confidence': conf_w}
                print(f"[predict-sign] words label={label_w} confidence={conf_w:.4f} probs={np.round(preds_w, 4).tolist()}")

        # Alphabet model prediction
        if sign_model_alpha is not None:
            preds_a = predict_ensemble(sign_model_alpha)
            if preds_a is not None:
                class_index_a = int(np.argmax(preds_a))
                conf_a = float(np.max(preds_a))
                label_a = SIGN_LABELS_ALPHA.get(class_index_a, 'Unknown')
                response['alphabet'] = {'label': label_a, 'confidence': conf_a}
                print(f"[predict-sign] alphabet label={label_a} confidence={conf_a:.4f} probs={np.round(preds_a, 4).tolist()}")

        # Choose a single label to return as `label` for backward compatibility.
        # Preference: highest confidence across available models, provided it exceeds threshold.
        chosen = {'label': 'Detecting...', 'confidence': 0.0}
        threshold = 0.55
        for k in ('words', 'alphabet'):
            v = response.get(k)
            if v and v['confidence'] > chosen['confidence']:
                chosen = {'label': v['label'], 'confidence': v['confidence'], 'source': k}

        state = _get_prediction_state()
        if chosen['confidence'] >= threshold and chosen['label'] != 'Detecting...':
            state['buffer'].append((chosen['label'], chosen['confidence']))
            if len(state['buffer']) >= 3:
                label_counts = {}
                label_confidences = {}
                for label, confidence in state['buffer']:
                    label_counts[label] = label_counts.get(label, 0) + 1
                    label_confidences[label] = label_confidences.get(label, 0.0) + confidence
                most_common_label = max(
                    label_counts.items(),
                    key=lambda item: (item[1], label_confidences[item[0]] / item[1]),
                )[0]
                if label_counts[most_common_label] >= 3:
                    state['stable_label'] = most_common_label
                    state['stable_confidence'] = label_confidences[most_common_label] / label_counts[most_common_label]
        else:
            state['buffer'].clear()

        if state['stable_label'] != 'Detecting...' and state['stable_confidence'] >= threshold:
            response['label'] = state['stable_label']
            response['confidence'] = state['stable_confidence']
            response['source'] = chosen.get('source')
        elif chosen['confidence'] >= threshold:
            response['label'] = chosen['label']
            response['confidence'] = chosen['confidence']
            response['source'] = chosen.get('source')
        else:
            response['label'] = 'Detecting...'
            response['confidence'] = 0.0

        return jsonify(response)
    except Exception as e:
        app.logger.exception('Prediction error: %s', e)
        return jsonify({"error": "Prediction failed. Please try again."}), 500


@app.route('/restore-trash', methods=['POST'])
@login_required
def restore_trash():
    data = request.get_json() or {}
    trash_id = data.get('trash_id')
    username = session.get('username')
    if trash_id is None:
        return jsonify({"error": "missing trash_id"}), 400

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, orig_id, username, text, created_at FROM history_trash WHERE id = ? AND username = ?', (trash_id, username))
        row = cur.fetchone()
        if not row:
            conn.close()
            return jsonify({"restored": False}), 200
        _, orig_id, uname, text, created_at = row
        cur.execute('INSERT INTO history (username, text, created_at) VALUES (?, ?, ?)', (uname, text, created_at))
        new_id = cur.lastrowid
        cur.execute('DELETE FROM history_trash WHERE id = ? AND username = ?', (trash_id, username))
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500
    conn.close()
    return jsonify({"restored": True, "new_id": new_id})


@app.route('/restore-last', methods=['POST'])
@login_required
def restore_last():
    username = session.get('username')
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, orig_id, username, text, created_at FROM history_trash WHERE username = ? ORDER BY deleted_at DESC LIMIT 1', (username,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return jsonify({"restored": False}), 200
        trash_id, orig_id, uname, text, created_at = row
        cur.execute('INSERT INTO history (username, text, created_at) VALUES (?, ?, ?)', (uname, text, created_at))
        new_id = cur.lastrowid
        cur.execute('DELETE FROM history_trash WHERE id = ? AND username = ?', (trash_id, username))
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500
    conn.close()
    return jsonify({"restored": True, "new_id": new_id})


# =========================
# RUN APP
# =========================
@app.route("/translate-page")
@login_required
def translate_page():

    return render_template(
        "translate.html"
    )


@app.route("/translate", methods=["POST"])
@login_required
def translate():

    try:

        data = request.get_json()

        text = data.get("text")

        language = data.get("language")

        translator = Translator()

        translated = translator.translate(
            text,
            dest=language
        )

        print("Translated:", translated.text)

        return jsonify({
            "translated_text": translated.text
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "translated_text":
            "Translation Error"
        })



@app.route("/speak", methods=["POST"])
@login_required
def speak():

    data = request.get_json()

    text = data.get("text", "")
    language = data.get("language", "en")

    if not text.strip():
        return jsonify({"error": "Text is required."}), 400

    translator = Translator()
    translated = translator.translate(
        text,
        dest=language
    )

    translated_text = translated.text

    username = session['username']
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (username, text) VALUES (?, ?)",
        (username, translated_text)
    )
    conn.commit()
    conn.close()

    audio_filename = f"output_{uuid4().hex}.mp3"
    static_dir = os.path.join(os.getcwd(), 'static')
    audio_path = os.path.join(static_dir, audio_filename)

    tts = gTTS(
        text=translated_text,
        lang=language
    )

    # Ensure `static` directory exists before saving audio
    try:
        os.makedirs(static_dir, exist_ok=True)
        tts.save(audio_path)

        return jsonify({
            "audio": url_for('static', filename=audio_filename)
        })
    except Exception as e:
        print('TTS save error:', e)
        return jsonify({"error": f"TTS save failed: {e}"}), 500

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )