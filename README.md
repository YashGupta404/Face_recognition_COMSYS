# 🤖 Face Recognition & Gender Classification (FACECOM)

A complete deep learning pipeline built for the **FACECOM 2025 Challenge**, supporting:

- 👩‍🦰 **Task A:** Gender Classification  
- 🧑‍💻 **Task B:** Face Recognition under Distorted Conditions

> ✅ All models are hosted on Hugging Face Hub – no manual downloads required.  
> Just place your test folder, run the script, and get evaluation metrics instantly!

---

## 📁 Project Structure

```
Face_recognition_COMSYS/
├── train_a/                         # Task A - Gender Classification
│   ├── test.py                      # Evaluation script
│   ├── Task_A.ipynb                 # Training notebook
│   └── requirements.txt
│
├── train_b/                         # Task B - Face Recognition
│   ├── predict_test.py              # Evaluation script
│   ├── Task_B.ipynb                 # Training notebook
│   └── requirements.txt
│
├── report.pdf                       # Technical Report
├── README.md
├── .gitignore
└── .gitattributes
```

---

## 🧑‍🦰 TASK A – Gender Classification

### 📐 Model Architecture

- ✅ **Transfer Learning:** VGG16 / ResNet50 backbone
- ✅ **Binary Classification:** Male vs Female
- ✅ **Input Shape:** 150x150 RGB
- ✅ **Optimizer:** Adam
- ✅ **Callbacks:** EarlyStopping, ReduceLROnPlateau
- ✅ **Data Augmentation:** Rotation, Flip, Brightness, Zoom

### 📂 Expected Test Folder Structure

```
test/
├── male/
│   ├── image1.jpg
│   └── ...
└── female/
    ├── image2.jpg
    └── ...
```

### 🔧 How to Run

```bash
cd train_a
pip install -r requirements.txt
# ⬇️ Make sure to update the TEST_DIR variable inside test.py with your test folder path
python test.py
```

✅ This Will:

- 🔁 Automatically download model & preprocessor from Hugging Face  
- 📊 Output metrics (Accuracy, Precision, Recall, F1 Score)  
- 💾 Save metrics to `test_metrics.json`  

---

## 🧠 TASK B – Face Recognition under Distorted Conditions

### 📐 Model Architecture

- ✅ **Backbones:** EfficientNetB3, ResNet50, MobileNetV2
- ✅ **Loss:** ArcFace (metric learning for angular separation)
- ✅ **Optimizer:** Sharpness-Aware Minimization (SAM) + Cosine LR
- ✅ **Input Shape:** 96x96 RGB
- ✅ **Ensemble Voting** via embedding distances (k-NN)

### 📂 Expected Test Folder Structure

```
test/
├── 001_frontal/
│   ├── 001_frontal.jpg
│   └── distortion/
│       ├── blur.jpg
│       └── lowlight.jpg
├── 002_frontal/
│   ├── 002_frontal.jpg
│   └── distortion/
│       ├── fog.jpg
│       └── ...
```

### 🔧 How to Run

```bash
cd train_b
pip install -r requirements.txt
# ⬇️ Make sure to update the TEST_FOLDER variable inside predict_test.py with your test folder path
python predict_test.py
```

✅ This Will:

- 📦 Automatically load model weights & LabelEncoder from Hugging Face  
- 📊 Evaluate Top-1 accuracy, Precision, Recall, F1  
- 💾 Save metrics to `test_metrics.json`

---

## ☁️ Model Hosting on Hugging Face

All models are publicly hosted on:

👉 https://huggingface.co/YashGupta05/facecom-task-b-weights

**Hosted Files:**

- `Task_A_.keras` — Gender classifier
- `preprocessor_task_A.pkl` — Preprocessing config for Task A
- `model.keras` — Face recognition model
- `preprocess.pkl` — LabelEncoder for Task B

No manual download needed. Scripts handle loading automatically.

---

## 💡 Highlights

- ✅ No hardcoded filenames or manual downloads  
- 📦 Fully automatic Hugging Face model loading  
- 🧠 ArcFace + Transfer Learning architectures  
- ⚡ Fast, GPU-optimized execution on Colab  
- 🧪 Ready for real-world distortion conditions

---

## 🤝 Contributors

- **Yash Gupta** – Developed Task B  
- **Riti Kant Juhi** – Developed Task A  
- **Rohit Roy** – Wrote Technical Report

---

## 📜 License

Licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
