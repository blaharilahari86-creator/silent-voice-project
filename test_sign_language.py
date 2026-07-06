#!/usr/bin/env python3
"""
Comprehensive test suite for sign language feature
"""
import os
import sys
import json
import base64
import sqlite3
import numpy as np
import cv2
from app import app, sign_model, SIGN_LABELS

print("=" * 80)
print("SIGN LANGUAGE FEATURE TEST SUITE")
print("=" * 80)

# Test 1: Check if model is loaded
print("\n[TEST 1] Model Loading")
print("-" * 80)
if sign_model is not None:
    print("✓ Sign model loaded successfully")
    print(f"  Model type: {type(sign_model)}")
    print(f"  Model summary:\n{sign_model.summary()}")
else:
    print("✗ FAILED: Sign model not loaded")
    sys.exit(1)

# Test 2: Check SIGN_LABELS
print("\n[TEST 2] Sign Labels Configuration")
print("-" * 80)
if SIGN_LABELS:
    print(f"✓ Sign labels loaded: {len(SIGN_LABELS)} labels")
    for idx, label in SIGN_LABELS.items():
        print(f"  [{idx}] {label}")
else:
    print("✗ FAILED: No sign labels found")
    sys.exit(1)

# Test 3: Verify endpoints exist
print("\n[TEST 3] Flask Routes")
print("-" * 80)
routes_to_check = ['/sign-language', '/predict-sign', '/capture-sample']
found_routes = []
for rule in app.url_map.iter_rules():
    if any(route in rule.rule for route in routes_to_check):
        found_routes.append(rule.rule)
        print(f"✓ Found route: {rule.rule} {list(rule.methods)}")

missing = [r for r in routes_to_check if not any(r in fr for fr in found_routes)]
if missing:
    print(f"✗ FAILED: Missing routes: {missing}")
else:
    print("✓ All required routes found")

# Test 4: Test preprocessing functions
print("\n[TEST 4] Preprocessing Functions")
print("-" * 80)
try:
    from preprocess import prepare_variants, to_model_input
    print("✓ Import successful")
    
    # Create a dummy image
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    variants = prepare_variants(dummy_img)
    print(f"✓ prepare_variants() works: {list(variants.keys())}")
    
    for name, var in variants.items():
        inp = to_model_input(var)
        print(f"  - {name}: shape {var.shape} -> model input shape {inp.shape}")
        if inp.shape != (1, 28, 28, 1):
            print(f"✗ FAILED: Expected shape (1, 28, 28, 1), got {inp.shape}")
            sys.exit(1)
    print("✓ Model input shapes are correct")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Test model prediction
print("\n[TEST 5] Model Prediction")
print("-" * 80)
try:
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    variants = prepare_variants(dummy_img)
    
    predictions = []
    for name, var in variants.items():
        inp = to_model_input(var)
        preds = sign_model.predict(inp, verbose=0)[0]
        confidence = float(np.max(preds))
        class_index = int(np.argmax(preds))
        label = SIGN_LABELS.get(class_index, 'Unknown')
        predictions.append({
            'variant': name,
            'label': label,
            'confidence': confidence,
            'class_index': class_index
        })
        print(f"  {name:12} -> {label:15} (confidence: {confidence:.4f})")
    
    print("✓ Model predictions successful")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test Flask endpoints with mock client
print("\n[TEST 6] Flask Endpoints")
print("-" * 80)

client = app.test_client()

# Create test user
username = 'test_sign_user'
password = 'testpass123'

# Clean up previous test user
try:
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE username = ?", (username,))
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()
except:
    pass

# Register
print("  Testing registration...")
res = client.post('/register', json={'username': username, 'password': password})
if res.status_code in [200, 201]:
    print(f"  ✓ Registration successful (status: {res.status_code})")
else:
    print(f"  ✗ Registration failed (status: {res.status_code}): {res.get_json()}")

# Login
print("  Testing login...")
res = client.post('/login', json={'username': username, 'password': password})
if res.status_code in [200, 201]:
    print(f"  ✓ Login successful (status: {res.status_code})")
else:
    print(f"  ✗ Login failed (status: {res.status_code}): {res.get_json()}")
    sys.exit(1)

# Create a valid image (small random image encoded as PNG)
print("  Creating test image...")
test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
_, png_bytes = cv2.imencode('.png', test_img)
image_b64 = base64.b64encode(png_bytes).decode('utf-8')
image_data = f'data:image/png;base64,{image_b64}'

# Test predict-sign endpoint
print("  Testing /predict-sign endpoint...")
try:
    res = client.post('/predict-sign', 
                     json={'image': image_data},
                     content_type='application/json')
    
    if res.status_code == 200:
        result = res.get_json()
        label = result.get('label')
        confidence = result.get('confidence')
        print(f"  ✓ Prediction successful")
        print(f"    - Label: {label}")
        print(f"    - Confidence: {confidence}")
    else:
        print(f"  ✗ Prediction failed (status: {res.status_code})")
        print(f"    Response: {res.get_json()}")
except Exception as e:
    print(f"  ✗ Error during prediction: {e}")

# Test capture-sample endpoint
print("  Testing /capture-sample endpoint...")
try:
    sample_label = list(SIGN_LABELS.values())[0] if SIGN_LABELS else 'ALL_DONE'
    res = client.post('/capture-sample',
                     json={'image': image_data, 'label': sample_label},
                     content_type='application/json')
    
    if res.status_code == 200:
        result = res.get_json()
        print(f"  ✓ Sample capture successful")
        print(f"    Message: {result.get('message')}")
        print(f"    Path: {result.get('path')}")
    else:
        print(f"  ✗ Sample capture failed (status: {res.status_code})")
        print(f"    Response: {res.get_json()}")
except Exception as e:
    print(f"  ✗ Error during capture: {e}")

# Test 7: Check database integration
print("\n[TEST 7] Database Integration")
print("-" * 80)
try:
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    # Check if user exists
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user:
        print(f"✓ User found in database: {user}")
    else:
        print("✗ User not found in database")
    
    # Check history table schema
    c.execute("PRAGMA table_info(history)")
    columns = {row[1]: row[2] for row in c.fetchall()}
    required_cols = ['id', 'username', 'text', 'created_at']
    for col in required_cols:
        if col in columns:
            print(f"✓ Column '{col}' exists in history table ({columns[col]})")
        else:
            print(f"✗ Column '{col}' missing from history table")
    
    conn.close()
except Exception as e:
    print(f"✗ Database check failed: {e}")

# Test 8: Sign language HTML template
print("\n[TEST 8] Sign Language Template")
print("-" * 80)
template_path = os.path.join(os.getcwd(), 'templates', 'sign_language.html')
if os.path.exists(template_path):
    with open(template_path, 'r') as f:
        content = f.read()
    
    required_elements = [
        'startCamera',
        'stopCamera',
        'captureAndPredict',
        '/predict-sign',
        '/capture-sample',
        'video'
    ]
    
    missing = []
    for elem in required_elements:
        if elem not in content:
            missing.append(elem)
        else:
            print(f"✓ Found '{elem}' in template")
    
    if missing:
        print(f"✗ Missing elements: {missing}")
    else:
        print("✓ All required elements found in template")
else:
    print(f"✗ Template not found at {template_path}")

# Test 9: File structure check
print("\n[TEST 9] Project File Structure")
print("-" * 80)
required_files = [
    'sign_model.h5',
    'sign_labels.py',
    'hand_detect.py',
    'predict_sign.py',
    'preprocess.py',
    'gesture_text.py',
    'templates/sign_language.html',
    'dataset/real_capture'
]

for file_path in required_files:
    full_path = os.path.join(os.getcwd(), file_path)
    if os.path.exists(full_path):
        if os.path.isdir(full_path):
            print(f"✓ Directory exists: {file_path}")
        else:
            size = os.path.getsize(full_path)
            print(f"✓ File exists: {file_path} ({size} bytes)")
    else:
        print(f"✗ Missing: {file_path}")

# Cleanup
print("\n[CLEANUP]")
print("-" * 80)
try:
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE username = ?", (username,))
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    print("✓ Test user cleaned up")
except Exception as e:
    print(f"✗ Cleanup failed: {e}")

print("\n" + "=" * 80)
print("TEST SUITE COMPLETED")
print("=" * 80)
