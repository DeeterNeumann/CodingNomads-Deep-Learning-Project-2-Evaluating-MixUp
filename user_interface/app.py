import os
import sys

import streamlit as st
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets as tvds, transforms as T
from PIL import Image

# allows importing from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from resnet_miniproject_core import ImageClassifier, AnyImageDM, load_dataset_any

# device and checkpoint paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# image transforms (ResNet / ImageNet-Style)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR100_ROOT = os.path.join(PROJECT_ROOT, "data")

_cifar100 = tvds.CIFAR100(
    root=CIFAR100_ROOT,
    train=False,
    download=False
)
CIFAR100_CLASSES = _cifar100.classes

# model loading helper
@st.cache_resource
def load_model(dataset_name: str, use_mixup: bool) -> ImageClassifier:
    """
    Load a Lightning checkpoint for a given dataset and mixup setting.

    dataset_name: "cifar10" or "cifar100"
    use_mixup: False -> *_nomix.ckpt, True -> *._mixup.ckpt
    """
    dataset_name = dataset_name.lower()
    mix_suffix = "mixup" if use_mixup else "nomix"

    ckpt_name = f"{dataset_name}_{mix_suffix}.ckpt"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Lightning will reconstruct ImageClassifier with saved hyperparameters
    model = ImageClassifier.load_from_checkpoint(ckpt_path)

    model.to(DEVICE)
    model.eval()
    return model

# image preprocessing helper
def preprocess_image(img: Image.Image) -> torch.Tensor:
    """
    Convert uploaded PIL image into model-ready tensor
    Ensures RGB, resizes to 224x224, normalizes with ImageNet stats.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    tensor = transform(img) # shape: C x H x W
    return tensor

# prediction helper
def predict(model: ImageClassifier, img_tensor: torch.Tensor):
    """
    Run a single image through the model.

    Returns:
        pred_idx    (int): predicted class index
        conf        (float): max softmax probability
        probs       (Tensor): full probability vector over classes
    """
    model.eval()
    img_batch = img_tensor.unsqueeze(0).to(DEVICE) # (1, C, H, W)

    with torch.no_grad():
        logits = model(img_batch)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0)
        conf, pred_idx = torch.max(probs, dim=0)

    return int(pred_idx), float(conf), probs

# ------
# DATASETS = {
#     "CIFAR10": {
#         "ckpt_no_mixup": "cifar10_nomix.ckpt",
#         "ckpt_mixup": "cifar10_mixup.ckpt",
#         ...
#     },
#     "CIFAR100": {
#         "ckpt_no_mixup": "cifar100_nomix.ckpt",
#         "ckpt_mixup": "cifar100_mixup.ckpt",
#         ...
#     },
# }

# ckpt_path = os.path.join(os.path.dirname(__file__), cfg["ckpt_no_mixup"])

# ckpt_path = os.path.join(os.path.dirname(__file__), cfg["ckpt_mixup"])
# ------

# Streamlit UI

st.set_page_config(page_title="CIFAR10 & CIFAR100 Mixup Demo", layout="wide")
st.title("CIFAR10 & CIFAR100 - Mixup vs. No Mixup")

st.write(
    "Upload an image and compare predictions from four models:\n"
    "- **CIFAR10 (no mixup)**\n"
    "- **CIFAR10 (mixup)**\n"
    "- **CIFAR100 (no mixup)**\n"
    "- **CIFAR100 (mixup)**"
)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # show original image
    pil_img = Image.open(uploaded_file)
    st.subheader("Input image")
    st.image(pil_img, use_container_width=False)

    # preprocess once and reuse for all four models
    img_tensor = preprocess_image(pil_img)

    st.subheader("Model predictions (2x2 matrix)")

    def show_model_result(dataset_name: str, use_mixup: bool, container):
        model = load_model(dataset_name, use_mixup)
        pred_idx, conf, probs = predict(model, img_tensor)

        if dataset_name.lower() == "cifar10":
            label = CIFAR10_CLASSES[pred_idx]
        else:
            label = CIFAR100_CLASSES[pred_idx]

        mix_label = "with mixup" if use_mixup else "no mixup"
        title = f"{dataset_name.upper()} - {mix_label}"

        with container:
            st.markdown(f"### {title}")
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** {conf*100:.2f}%")

    # Row 1: CIFAR10 (no mixup | mixup)
    col1, col2 = st.columns(2)
    show_model_result("cifar10", False, col1)
    show_model_result("cifar10", True, col2)

    # Row 2: CIFAR100 (no mixup | mixup)
    col3, col4 = st.columns(2)
    show_model_result("cifar100", False, col3)
    show_model_result("cifar100", True, col4)

else:
    st.info("Upload an image to see predictions from all four models.")

        