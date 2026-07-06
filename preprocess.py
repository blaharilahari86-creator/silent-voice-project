import cv2
import numpy as np


def _center_square_crop(img):
    h, w = img.shape[:2]
    side = min(h, w)
    startx = max(0, (w - side) // 2)
    starty = max(0, (h - side) // 2)
    return img[starty:starty + side, startx:startx + side]


def _safe_crop(img, box=None):
    if box is not None:
        x_min, y_min, x_max, y_max = box
        h, w = img.shape[:2]
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(1, min(x_max, w))
        y_max = max(1, min(y_max, h))
        crop = img[y_min:y_max, x_min:x_max]
        if crop.size > 0:
            return crop
    return _center_square_crop(img)


def prepare_gray(img, box=None, size=(28, 28)):
    """Match the training pipeline as closely as possible.

    The current model was trained on 28x28 grayscale images with straightforward
    normalization. Applying CLAHE, blur, and brightness variants during live
    inference changes the input distribution and hurts accuracy.
    """
    crop = _safe_crop(img, box)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


def prepare_variants(img, box=None, size=(28, 28)):
    base = prepare_gray(img, box, size=size)
    return {
        'orig': base,
        'flipped': cv2.flip(base, 1),
    }


def to_model_input(arr):
    a = arr.astype('float32') / 255.0
    return a.reshape(1, arr.shape[0], arr.shape[1], 1)
