import os
import sys
import types

# ✅ Fix for tensorflow_decision_forests issue (Windows safe)
sys.modules['tensorflow_decision_forests'] = types.ModuleType('tensorflow_decision_forests')

import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "mask_detector.h5"
OUTPUT_DIR = "web_model"

# ==============================
# CHECK MODEL
# ==============================
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found: {MODEL_PATH}")
    print("👉 Make sure 'mask_detector.h5' is in this folder")
    exit(1)

# ==============================
# LOAD MODEL
# ==============================
print("[INFO] Loading trained Keras model...")
model = load_model(MODEL_PATH, compile=False)

# ==============================
# CREATE OUTPUT FOLDER
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# CONVERT MODEL
# ==============================
print("[INFO] Converting model to TensorFlow.js format...")
tfjs.converters.save_keras_model(model, OUTPUT_DIR)

# ==============================
# SUCCESS
# ==============================
print("\n[SUCCESS] Model converted successfully! 🚀")
print(f"📁 Output folder: {OUTPUT_DIR}")
print("👉 Files generated:")
print("   - model.json")
print("   - group*.bin")