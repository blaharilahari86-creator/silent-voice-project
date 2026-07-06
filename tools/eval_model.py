import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Add parent directory to the path so we can import sign_labels
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sign_labels import SIGN_LABELS_WORDS, SIGN_LABELS_ALPHA


def resolve_csv(arg_path, env_var, default_path, keywords=None):
    if arg_path:
        return arg_path
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path
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
    return None


def load_csv(path):
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError('Test CSV must include a label column.')
    y = df['label'].values
    X = df.drop('label', axis=1).values.astype('float32') / 255.0
    X = X.reshape(-1, 28, 28, 1)
    return X, y


def detect_dataset_type(y_values):
    unique = sorted(np.unique(y_values))
    if len(unique) == len(SIGN_LABELS_WORDS) and set(unique) == set(range(len(SIGN_LABELS_WORDS))):
        return 'word', None
    if len(unique) == len(SIGN_LABELS_ALPHA):
        if set(unique) == set(range(len(SIGN_LABELS_ALPHA))):
            return 'alphabet', None
        missing_nine = set(range(len(SIGN_LABELS_ALPHA) + 1)) - {9}
        if set(unique) == missing_nine:
            return 'alphabet', unique
    if len(unique) == len(SIGN_LABELS_WORDS) and max(unique) <= len(SIGN_LABELS_WORDS) - 1:
        return 'word', None
    return None, None


def remap_labels(y_values, label_order):
    label_map = {orig: idx for idx, orig in enumerate(label_order)}
    return np.array([label_map[int(val)] for val in y_values], dtype=np.int32)


def evaluate_model(model_path, X_test, y_test, labels_map, label_order=None):
    if label_order is not None:
        y_test = remap_labels(y_test, label_order)

    model = load_model(model_path)
    preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    acc = np.mean(y_pred == y_test)
    print(f'Overall accuracy: {acc:.4f} ({int(acc*100)}%)')

    print('\nPer-class accuracy:')
    for cls in sorted(np.unique(y_test)):
        mask = y_test == cls
        cls_acc = np.mean(y_pred[mask] == cls)
        label = labels_map.get(int(cls), f'Class {cls}')
        print(f'  {label:15} ({int(cls):2}) : {cls_acc:.4f} ({int(cls_acc*100):3}%)')


def main():
    parser = argparse.ArgumentParser(description='Evaluate sign language models')
    parser.add_argument('--model', default='both', choices=['word', 'alphabet', 'both'], help='Which model to evaluate')
    parser.add_argument('--test-csv', help='Path to test CSV')
    args = parser.parse_args()

    base = os.getcwd()
    model_word_path = os.path.join(base, 'sign_model.h5')
    model_alpha_path = os.path.join(base, 'sign_model_alpha.h5')
    default_test = os.path.join(base, 'dataset', 'archive', 'sign_mnist_test.csv')
    test_csv = resolve_csv(args.test_csv, 'SIGN_MNIST_TEST_CSV', default_test, keywords=['test', 'sign'])

    if not test_csv or not os.path.exists(test_csv):
        print('Test CSV not found at', test_csv)
        return

    print('Loading test data...')
    X_test, y_test = load_csv(test_csv)

    dataset_type, label_order = detect_dataset_type(y_test)
    print('Detected dataset type:', dataset_type)
    if dataset_type is None:
        print('Could not detect dataset type from the test labels. Unique label values:', sorted(np.unique(y_test)))
        return

    if args.model in ['word', 'both'] and dataset_type == 'alphabet':
        print('Skipping word model evaluation because the test set is alphabet labels.')
    if args.model in ['alphabet', 'both'] and dataset_type == 'word':
        print('Skipping alphabet model evaluation because the test set is word labels.')

    if args.model in ['word', 'both'] and dataset_type == 'word':
        if os.path.exists(model_word_path):
            print('\n' + '=' * 80)
            print('Evaluating WORD model:', model_word_path)
            print('=' * 80)
            evaluate_model(model_word_path, X_test, y_test, SIGN_LABELS_WORDS)
        else:
            print('Word model not found at', model_word_path)

    if args.model in ['alphabet', 'both'] and dataset_type == 'alphabet':
        if os.path.exists(model_alpha_path):
            print('\n' + '=' * 80)
