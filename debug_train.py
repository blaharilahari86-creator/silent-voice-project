#!/usr/bin/env python3
import os
import pandas as pd
from sign_labels import ASL_ALPHABET_LABELS, WORD_LABELS

# Test dataset loading
csv_path = 'dataset/archive/sign_mnist_train.csv'
print(f"Loading dataset from {csv_path}")
df = pd.read_csv(csv_path)

print(f"Dataset rows: {len(df)}")
print(f"Unique labels: {sorted(df['label'].unique())}")
print(f"Label count: {df['label'].nunique()}")
print(f"ASL_ALPHABET_LABELS count: {len(ASL_ALPHABET_LABELS)}")
print(f"WORD_LABELS count: {len(WORD_LABELS)}")

# Test label detection logic
unique_values = sorted(df['label'].unique().tolist())
print(f"\nLabel detection logic:")
if len(unique_values) == len(ASL_ALPHABET_LABELS):
    print(f"✓ Detected 24-class alphabet dataset")
    model_labels = [ASL_ALPHABET_LABELS[i] for i in range(len(unique_values))]
    print(f"Mapped labels: {model_labels}")
elif len(unique_values) == len(WORD_LABELS):
    print(f"Detected 8-class word dataset")
else:
    print(f"No match for class count {len(unique_values)}")

# Test training script execution
print(f"\nRunning training with minimal epochs...")
os.system("python train_model.py --train-csv 'dataset/archive/sign_mnist_train.csv' --save-model 'sign_model_alpha.h5' --save-label-map 'sign_label_map_alpha.json' --epochs 1")

# Check if file was created
if os.path.exists('sign_model_alpha.h5'):
    print("\n✓ sign_model_alpha.h5 created successfully!")
    print(f"File size: {os.path.getsize('sign_model_alpha.h5')} bytes")
else:
    print("\n✗ sign_model_alpha.h5 was NOT created")
    
if os.path.exists('sign_label_map_alpha.json'):
    print("✓ sign_label_map_alpha.json created successfully!")
    with open('sign_label_map_alpha.json', 'r') as f:
        print(f.read())
else:
    print("✗ sign_label_map_alpha.json was NOT created")
