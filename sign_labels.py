"""Centralized sign labels mapping.

The default mapping uses the standard Sign Language MNIST alphabet classes.
If a training run writes a dynamic label map file, it is loaded automatically
so the web app and inference scripts remain aligned with the trained model.
"""

import json
import os

ASL_ALPHABET_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y'
]

WORD_LABELS = [
    'ALL_DONE',
    'YES',
    'HELP',
    'PLEASE',
    'MORE',
    'THANK_YOU',
    'NO',
    'STOP'
]

# The current app and model use the word-based sign set by default.
# If a valid sign_label_map.json exists, it is loaded and kept in sync.
DEFAULT_LABELS = WORD_LABELS


def load_all_label_maps(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.getcwd(), 'sign_label_map.json')
    
    alpha_path = os.path.join(os.getcwd(), 'sign_label_map_alpha.json')

    words_map = {i: label for i, label in enumerate(WORD_LABELS)}
    alpha_map = {i: label for i, label in enumerate(ASL_ALPHABET_LABELS)}

    # Try to load word label map
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Support structured maps: {"words": [...], "alphabet": [...]} or legacy {"labels": [...]}
                if 'words' in data and isinstance(data['words'], list):
                    words_map = {i: label for i, label in enumerate(data['words'])}
                if 'alphabet' in data and isinstance(data['alphabet'], list):
                    alpha_map = {i: label for i, label in enumerate(data['alphabet'])}
                if 'labels' in data and isinstance(data['labels'], list):
                    # legacy single-list file — assume it maps to words when length matches
                    labels = data['labels']
                    if len(labels) == len(WORD_LABELS):
                        words_map = {i: label for i, label in enumerate(labels)}
                    elif len(labels) == len(ASL_ALPHABET_LABELS):
                        alpha_map = {i: label for i, label in enumerate(labels)}
        except Exception:
            pass
    
    # Try to load alphabet label map (dedicated file takes priority)
    if os.path.exists(alpha_path):
        try:
            with open(alpha_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'labels' in data and isinstance(data['labels'], list):
                alpha_map = {i: label for i, label in enumerate(data['labels'])}
        except Exception:
            pass

    return words_map, alpha_map


# Load both maps; keep backwards compatibility with SIGN_LABELS as the primary (words) map.
SIGN_LABELS_WORDS, SIGN_LABELS_ALPHA = load_all_label_maps()
SIGN_LABELS = SIGN_LABELS_WORDS

LABEL_TO_INDEX = {label: i for i, label in SIGN_LABELS.items()}


def get_label_map(kind='words'):
    if kind == 'alphabet':
        return SIGN_LABELS_ALPHA
    return SIGN_LABELS_WORDS


def num_classes(kind='words'):
    return len(get_label_map(kind))
