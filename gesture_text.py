import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
from collections import deque

from preprocess import prepare_variants, to_model_input
from sign_labels import SIGN_LABELS

model = load_model("sign_model.h5")
engine = pyttsx3.init()
labels = SIGN_LABELS

cap = cv2.VideoCapture(0)
last_label = ""
prediction_buffer = deque(maxlen=10)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    box_size = min(w, h) * 0.5
    x1 = int((w - box_size) // 2)
    y1 = int((h - box_size) // 2)
    x2 = int(x1 + box_size)
    y2 = int(y1 + box_size)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    variants = prepare_variants(frame, box=(x1, y1, x2, y2))
    best_conf = 0.0
    best_class = None

    for var in variants.values():
        inp = to_model_input(var)
        preds = model.predict(inp, verbose=0)[0]
        conf = float(np.max(preds))
        cls = int(np.argmax(preds))
        if conf > best_conf:
            best_conf = conf
            best_class = cls

    if best_conf >= 0.65 and best_class in labels:
        prediction_buffer.append(labels[best_class])
    else:
        prediction_buffer.clear()

    if len(prediction_buffer) >= 5:
        label = max(set(prediction_buffer), key=prediction_buffer.count)
    else:
        label = "Detecting..."

    if label != last_label and label != "Detecting...":
        engine.say(f"Detected {label}")
        engine.runAndWait()
        last_label = label

    cv2.putText(frame, f'Prediction: {label}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Confidence: {best_conf:.2f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()