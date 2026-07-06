import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd

from sign_labels import SIGN_LABELS, LABEL_TO_INDEX, ASL_ALPHABET_LABELS, WORD_LABELS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def resolve_csv(arg_path, env_var, default_paths, keywords=None):
    if arg_path:
        # Remove surrounding quotes if present
        arg_path = arg_path.strip("'\"")
        return arg_path
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path
    for default_path in default_paths:
        if os.path.exists(default_path):
            return default_path
    candidates = glob.glob(os.path.join('dataset', '**', '*.csv'), recursive=True)
    if keywords:
        for c in candidates:
            low = c.lower()
            if any(k in low for k in keywords):
                return c
    if candidates:
        return candidates[0]
    raise FileNotFoundError('No dataset CSV found; checked defaults and dataset folder')


def load_real_capture_csv(root_dir, label_to_index, out_csv):
    rows = []
    for label_name, idx in label_to_index.items():
        folder = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            rows.append([idx] + img.flatten().tolist())
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['label'] + [f'pixel{i}' for i in range(28 * 28)])
    df.to_csv(out_csv, index=False)
    return df


def save_label_map(labels, output_path='sign_label_map.json'):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'labels': labels}, f, indent=2)


def compute_class_weights(y_values):
    classes, counts = np.unique(y_values, return_counts=True)
    total = len(y_values)
    return {int(cls): float(total) / (len(classes) * count) for cls, count in zip(classes, counts)}


parser = argparse.ArgumentParser(description='Train sign language model')
parser.add_argument('--train-csv', help='Path to train CSV file')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--save-model', default='sign_model.h5')
parser.add_argument('--save-label-map', default='sign_label_map.json')
args = parser.parse_args()

custom_csv = os.path.join('dataset', 'archive', 'custom_sign_words.csv')
sign_mnist_csv = os.path.join('dataset', 'archive', 'sign_mnist_train.csv')
real_capture_csv = os.path.join('dataset', 'archive', 'custom_real_capture.csv')

train_csv = resolve_csv(args.train_csv, 'SIGN_MNIST_TRAIN_CSV', [custom_csv, sign_mnist_csv], keywords=['custom', 'sign', 'word'])

if train_csv == custom_csv and not os.path.exists(custom_csv):
    if os.path.exists(real_capture_csv):
        print('Using real capture dataset CSV:', real_capture_csv)
        train_csv = real_capture_csv
    elif os.path.isdir(os.path.join('dataset', 'real_capture')):
        print('Generating CSV from dataset/real_capture folder...')
        os.makedirs(os.path.dirname(real_capture_csv), exist_ok=True)
        generated = load_real_capture_csv(os.path.join('dataset', 'real_capture'), LABEL_TO_INDEX, real_capture_csv)
        if generated is None:
            raise FileNotFoundError('No real capture samples found in dataset/real_capture')
        train_df = generated
        train_csv = real_capture_csv
        print('Generated training dataset:', train_csv)
    else:
        raise FileNotFoundError('No training dataset found; create dataset/archive/custom_sign_words.csv or capture real samples.')

if 'train_df' not in locals():
    print('Using training dataset:', train_csv)
    train_df = pd.read_csv(train_csv)

# LABELS

if 'label' not in train_df.columns:
    raise ValueError('Training CSV must contain a "label" column')

raw_labels = train_df['label']
if raw_labels.dtype == object:
    raw_labels = raw_labels.astype(str).str.strip()

if raw_labels.dtype == object and not all(label.isdigit() for label in raw_labels.unique()):
    unique_raw = sorted(raw_labels.unique())
    if all(label in LABEL_TO_INDEX for label in unique_raw):
        y_values = raw_labels.map(LABEL_TO_INDEX).astype(np.int32).to_numpy()
        model_labels = [label for label, idx in sorted(LABEL_TO_INDEX.items(), key=lambda item: item[1]) if label in unique_raw]
    else:
        label_map = {label: idx for idx, label in enumerate(unique_raw)}
        y_values = raw_labels.map(label_map).astype(np.int32).to_numpy()
        model_labels = unique_raw
else:
    y_values = raw_labels.astype(np.int32).to_numpy()
    unique_values = sorted(np.unique(y_values))
    
    # Auto-detect alphabet (24) vs words (8) based on class count
    # Handle non-contiguous indices (e.g., sign_mnist uses 0-8, 10-24, skipping 9)
    if len(unique_values) == len(ASL_ALPHABET_LABELS):
        # Map original indices to ASL alphabet labels in order
        model_labels = [ASL_ALPHABET_LABELS[i] for i in range(len(unique_values))]
        print(f'Detected ASL ALPHABET dataset (24 classes)')
    elif len(unique_values) == len(WORD_LABELS):
        model_labels = [WORD_LABELS[i] for i in range(len(unique_values))]
        print(f'Detected WORD LABELS dataset (8 classes)')
    else:
        model_labels = [str(val) for val in unique_values]

unique_values = sorted(np.unique(y_values))
if unique_values != list(range(len(unique_values))):
    print(f'Reindexing labels from non-contiguous indices: {unique_values}')
    label_map = {orig: idx for idx, orig in enumerate(unique_values)}
    y_values = np.array([label_map[val] for val in y_values], dtype=np.int32)
    # model_labels is already correct from the detection above

print('Detected classes:', len(unique_values))
print('Using label mapping:', dict(zip(range(len(model_labels)), model_labels)))
if len(unique_values) == len(ASL_ALPHABET_LABELS):
    print('Training ALPHABET model (24-class ASL)')
else:
    print(f'Training {len(unique_values)}-class model')

X = train_df.drop('label', axis=1).values.astype('float32') / 255.0
X = X.reshape(-1, 28, 28, 1)

from tensorflow.keras.utils import to_categorical

y = to_categorical(y_values, num_classes=len(unique_values))

# MODEL

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    )
)

model.add(
    MaxPooling2D(pool_size=(2,2))
)

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(
    MaxPooling2D(pool_size=(2,2))
)

model.add(Flatten())

model.add(Dense(128, activation='relu'))

num_classes = y.shape[1]

model.add(Dense(num_classes, activation='softmax'))

# COMPILE

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN

model.fit(
    X,
    y,
    epochs=args.epochs,
    batch_size=32,
    validation_split=0.2
)

# SAVE

model.save(args.save_model)
save_label_map(model_labels, args.save_label_map)
print(f"Saved model to {args.save_model}")
print(f"Saved label map to {args.save_label_map}")

print("MODEL TRAINED SUCCESSFULLY")