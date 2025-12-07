# 🚀 CodingNomads Deep Learning Project 2  
# **Evaluating the Impact of MixUp on Image Classification Models**

This project evaluates the effect of **MixUp augmentation** on deep learning performance for image classification using the **CIFAR-10** and **CIFAR-100** datasets.  
Four models were trained:

1. **CIFAR-10 (No MixUp)**
2. **CIFAR-10 (With MixUp)**
3. **CIFAR-100 (No MixUp)**
4. **CIFAR-100 (With MixUp)**

Each model uses a **ResNet-18** backbone with pretrained ImageNet weights and was trained in **PyTorch Lightning**.

A **Streamlit user interface** allows any user to upload an image and instantly compare predictions from all four models in a **2×2 visual grid**.

---

## 📸 Project Highlights

### ✔️ Side-by-side model comparison  
Upload any image (PNG/JPG) and compare predictions from:

| CIFAR10 | CIFAR100 |
|--------|----------|
| No MixUp | No MixUp |
| MixUp | MixUp |

Each model’s **top-1 prediction and confidence** is displayed.

---

### ✔️ Local checkpoint loading  
Checkpoints are stored locally in:

```
checkpoints/
```

These are **not** pushed to GitHub (ignored via `.gitignore`).

---

### ✔️ Automatic CIFAR100 class-name loading  
The UI loads CIFAR-100 labels directly from your local dataset folder:

```
data/cifar-100-python/
```

No need to hard-code 100 classes.

---

## 🧠 What Is MixUp?

**MixUp** is a data augmentation technique introduced in 2018.  
It generates new samples by linearly combining two images and their labels:

```
x' = λx₁ + (1 − λ)x₂
y' = λy₁ + (1 − λ)y₂
```

Benefits include:

- Reduces overfitting  
- Improves robustness to noisy labels  
- Encourages models to behave linearly between samples  
- Often boosts generalization performance  

This project evaluates how MixUp influences classification confidence and accuracy across two benchmark datasets.

---

## 🏗️ Repository Structure

```
CodingNomadsDeepLearningProject2/
│
├── resnet_miniproject_core.py         # Lightning ImageClassifier model
├── user_interface/
│   ├── app.py                         # Streamlit UI (2x2 comparison)
│   └── __init__.py
│
├── checkpoints/                       # Model checkpoints (.ckpt, ignored by Git)
│   ├── cifar10_nomix.ckpt
│   ├── cifar10_mix.ckpt
│   ├── cifar100_nomix.ckpt
│   └── cifar100_mix.ckpt
│
├── data/                              # CIFAR10/100 extracted datasets
│   ├── cifar-10-batches-py/
│   └── cifar-100-python/
│
├── gpu_function.py
├── updated_resnet-mini-project.ipynb
├── resnet_mini_project.ipynb
└── README.md
```

---

## 🧩 Installation

### 1. Clone the repository

```bash
git clone https://github.com/DeeterNeumann/CodingNomads-Deep-Learning-Project-2-Evaluating-MixUp.git
cd CodingNomads-Deep-Learning-Project-2-Evaluating-MixUp
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Minimal installation:

```bash
pip install torch torchvision pytorch-lightning streamlit pillow
```

Or full reproducibility:

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Streamlit App

### Ensure:

- `checkpoints/` contains the 4 `.ckpt` model files  
- `data/` contains CIFAR10 + CIFAR100 extracted folders:

```
data/cifar-10-batches-py/
data/cifar-100-python/
```

### Run:

```bash
streamlit run user_interface/app.py
```

Upload an image to see the **2×2 prediction comparison grid**.

---

## 🔬 Model Training Overview

Models were trained with:

- PyTorch Lightning `LightningModule`
- ResNet-18 backbone
- ImageNet preprocessing (224×224 + normalization)
- CrossEntropyLoss
- CIFAR10 / CIFAR100 DataModules
- Optional MixUp augmentation

Checkpoints follow:

```
cifar10_mix.ckpt
cifar10_nomix.ckpt
cifar100_mix.ckpt
cifar100_nomix.ckpt
```

---

## 🚫 Files Not Included in Git

The `.gitignore` excludes:

- `data/`
- `.ckpt` files
- Lightning logs
- `__pycache__/`
- `.DS_Store`

This keeps the repo lightweight and within GitHub storage limits.

---

## ✨ Future Improvements

- Add top-5 probability bars  
- Add inference-time comparison  
- Grad-CAM visualization  
- Deploy to Streamlit Cloud or HuggingFace Spaces  
- Add REST API for programmatic access  

---

## 🙌 Acknowledgments

Built as part of **CodingNomads Deep Learning Certification**.  
Thanks to the PyTorch & Lightning communities for tools enabling rapid experimentation.
