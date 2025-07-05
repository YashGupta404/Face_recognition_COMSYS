import os
import sys
import subprocess
import json
import pickle
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Auto-install packages ===
requirements = """
tensorflow>=2.11.0
scikit-learn>=1.2.0
numpy>=1.22.0
huggingface_hub>=0.20.0
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])

# === Download from Hugging Face Hub ===
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(repo_id="YashGupta05/facecom-task-b-weights", filename="Task_A_.keras")
PREPROCESSOR_PATH = hf_hub_download(repo_id="YashGupta05/facecom-task-b-weights", filename="preprocessor_task_A.pkl")

# === Local test path ===
TEST_DIR = "./train_a/test"  # CHANGE THIS TO YOUR TEST FOLDER PATH
BATCH_SIZE = 32
IMG_SIZE = (150, 150)

# === Load Preprocessing Config ===
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocess_config = pickle.load(f)

test_datagen = ImageDataGenerator(**preprocess_config)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# === Load model and predict ===
model = load_model(MODEL_PATH)
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int)

report = classification_report(
    y_true,
    y_pred,
    output_dict=True,
    target_names=list(test_generator.class_indices.keys())
)

# === Extract metrics ===
accuracy = report['accuracy']
precision_macro = report['macro avg']['precision']
recall_macro = report['macro avg']['recall']
f1_macro = report['macro avg']['f1-score']

print("\n=== Evaluation Metrics ===")
print(f"Accuracy          : {accuracy:.2f}")
print(f"Precision (macro): {precision_macro:.2f}")
print(f"Recall (macro)   : {recall_macro:.2f}")
print(f"F1 Score (macro) : {f1_macro:.2f}")

metrics = {
    "accuracy": round(accuracy, 4),
    "precision_macro": round(precision_macro, 4),
    "recall_macro": round(recall_macro, 4),
    "f1_macro": round(f1_macro, 4)
}

with open("test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved to test_metrics.json")
