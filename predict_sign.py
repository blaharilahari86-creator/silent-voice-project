import cv2
import numpy as np
import mediapipe as mp

from tensorflow.keras.models import load_model
from preprocess import prepare_variants, to_model_input
from sign_labels import SIGN_LABELS

# LOAD MODEL

model = load_model("sign_model.h5")

# LABELS

labels = SIGN_LABELS

# MEDIAPIPE

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.50, min_tracking_confidence=0.50)
mp_draw = mp.solutions.drawing_utils

# CAMERA

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    text = 'Detecting...'
    best_conf = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min = max(min(x_list) - 20, 0)
            y_min = max(min(y_list) - 20, 0)
            x_max = min(max(x_list) + 20, w)
            y_max = min(max(y_list) + 20, h)

            variants = prepare_variants(frame, box=(x_min, y_min, x_max, y_max))
            for image_variant in variants.values():
                inp = to_model_input(image_variant)
                preds = model.predict(inp, verbose=0)[0]
                conf = float(np.max(preds))
                if conf > best_conf:
                    best_conf = conf
                    best_label = labels.get(int(np.argmax(preds)), 'Unknown')

            if best_conf >= 0.65:
                text = best_label
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Silent Voice", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()