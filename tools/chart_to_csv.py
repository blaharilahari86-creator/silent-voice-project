import csv
import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np

LABELS = [
    'ALL_DONE',
    'YES',
    'HELP',
    'PLEASE',
    'MORE',
    'THANK_YOU',
    'NO',
    'STOP'
]


def process_crop(img, size=(28, 28)):
    img = img.convert('L')
    img = ImageOps.autocontrast(img)
    resample = getattr(Image, 'Resampling', Image).LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    img = img.resize(size, resample)
    return np.array(img).flatten().tolist()


def augment_crop(img, size=(28, 28)):
    variants = []
    base = img.convert('L')
    variants.append(base)

    variants.append(base.transpose(Image.FLIP_LEFT_RIGHT))
    variants.append(base.rotate(5, expand=False, fillcolor=255))
    variants.append(base.rotate(-5, expand=False, fillcolor=255))
    variants.append(base.filter(ImageFilter.GaussianBlur(radius=1)))
    variants.append(ImageOps.invert(base))

    resample = getattr(Image, 'Resampling', Image).LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    augmented = []
    for v in variants:
        v = ImageOps.autocontrast(v)
        v = v.resize(size, resample)
        augmented.append(v)

    return augmented


def crop_grid(image_path, rows=2, cols=4):
    img = Image.open(image_path)
    w, h = img.size
    cell_w = w / cols
    cell_h = h / rows
    crops = []

    for idx, label in enumerate(LABELS):
        row = idx // cols
        col = idx % cols
        left = int(col * cell_w)
        upper = int(row * cell_h)
        right = int((col + 1) * cell_w)
        lower = int((row + 1) * cell_h)
        crop = img.crop((left, upper, right, lower))
        crops.append((label, crop))

    return crops


def main(image_path=None, out_csv=None, augment=True):
    if image_path is None:
        image_path = os.path.join('dataset', 'archive', 'amer_sign2.png')
    if out_csv is None:
        out_csv = os.path.join('dataset', 'archive', 'custom_sign_words.csv')

    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Chart image not found: {image_path}')

    rows = []
    crops = crop_grid(image_path)
    for label, crop in crops:
        if augment:
            variants = augment_crop(crop)
        else:
            variants = [crop]

        for var in variants:
            rows.append([label] + process_crop(var))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + [f'pixel{i}' for i in range(28 * 28)])
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows to {out_csv}')


if __name__ == '__main__':
    main()
