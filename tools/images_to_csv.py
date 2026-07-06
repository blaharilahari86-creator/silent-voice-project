import os
import csv
import cv2
import argparse
from sign_labels import LABEL_TO_INDEX

def process_image(path, size=(28,28)):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    flat = gray.flatten().tolist()
    return flat

def main(root_dir, out_csv):
    rows = []
    for label_name, idx in LABEL_TO_INDEX.items():
        folder = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            path = os.path.join(folder, fname)
            flat = process_image(path)
            if flat is None:
                continue
            rows.append([idx] + flat)

    # write CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # header: label + pixel0..pixel783
        header = ['label'] + [f'pixel{i}' for i in range(28*28)]
        writer.writerow(header)
        writer.writerows(rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert image folders into sign-mnist CSV')
    parser.add_argument('root_dir', help='Root folder containing subfolders named exactly as labels in sign_labels.py')
    parser.add_argument('--out', default='dataset/archive/custom_real_capture.csv', help='Output CSV path')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    main(args.root_dir, args.out)
    print('Wrote CSV to', args.out)
