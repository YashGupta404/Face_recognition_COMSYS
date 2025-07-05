import os
import sys
import subprocess
import requests

# Step 1: Auto-install required packages (quietly)
requirements = """tensorflow>=2.11.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.22.0
joblib>=1.2.0
dill>=0.3.6
requests>=2.25.0
"""

if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("Created requirements.txt")

print("Installing packages...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "-r", "requirements.txt"])
    print("All required packages installed.")
except subprocess.CalledProcessError as e:
    print("Package installation failed:", e)
    sys.exit(1)

# Step 2: Import packages
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import dill
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 3: Configuration
TEST_FOLDER = "./train_b/test"  # <-- Change this to your test folder path
IMG_SIZE = 96
BATCH_SIZE = 64

# Step 4: Download model + label encoder from Hugging Face
HF_REPO_URL = "https://huggingface.co/YashGupta05/facecom-task-b-weights/resolve/main/"
MODEL_FILENAME = "model.keras"
ENCODER_FILENAME = "preprocess.pkl"
MODEL_PATH = f"./train_b/model_weights/{MODEL_FILENAME}"
ENCODER_PATH = f"./train_b/model_weights/{ENCODER_FILENAME}"

os.makedirs("train_b/model_weights", exist_ok=True)

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url.split('/')[-1]} from Hugging Face...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print(f"{dest_path} already exists, skipping download.")

download_file(HF_REPO_URL + MODEL_FILENAME, MODEL_PATH)
download_file(HF_REPO_URL + ENCODER_FILENAME, ENCODER_PATH)

# Step 5: Load model and label encoder
print("Loading model and label encoder...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
except Exception as e:
    print("Error loading model or encoder:", e)
    sys.exit(1)

# Step 6: Extract test images
def extract_images_from_dir(base_dir):
    records = []
    for person in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person)
        if not os.path.isdir(person_path):
            continue
        clear = glob.glob(os.path.join(person_path, '*.jpg'))
        distorted = glob.glob(os.path.join(person_path, 'distortion', '*.jpg'))
        for img in clear + distorted:
            records.append({'image_path': img, 'person_id': person})
    return pd.DataFrame(records)

print("Parsing test images...")
test_df = extract_images_from_dir(TEST_FOLDER)

try:
    test_df['label'] = le.transform(test_df['person_id'])
except ValueError:
    print("Some test classes are unknown to the model. Filtering known classes...")
    test_df = test_df[test_df['person_id'].isin(le.classes_)]
    test_df['label'] = le.transform(test_df['person_id'])

# Step 7: Load and preprocess test images
def load_image(img_path, label):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

test_dataset = tf.data.Dataset.from_tensor_slices((test_df['image_path'], test_df['label']))
test_dataset = test_dataset.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Step 8: Predict
print("Running predictions...")
pred_probs = model.predict(test_dataset, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_df['label'].values

# Step 9: Evaluation
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')

print("\n📊 Evaluation Results:")
print(f"✅ Top-1 Accuracy           : {acc:.4f}")
print(f"✅ Precision (Macro)        : {prec:.4f}")
print(f"✅ Recall (Macro)           : {recall:.4f}")
print(f"✅ Macro-averaged F1 Score  : {f1:.4f}")
