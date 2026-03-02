import json
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.hf_utils import hf_download, hf_read_json

REPO_ID = "ash001/cats-dogs-transferlearning-cnn"

st.set_page_config(page_title="Cats vs Dogs", page_icon="🐱", layout="wide")
st.title("🐱🐶 Cats vs Dogs (Transfer Learning CNN)")
st.caption("Repo: [ash001/cats-dogs-transferlearning-cnn](https://huggingface.co/ash001/cats-dogs-transferlearning-cnn)")

cfg = hf_read_json(REPO_ID, "artifacts/config.json")
image_size = int(cfg["image_size"])
mean = cfg["mean"]
std = cfg["std"]

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    class_names_path = hf_download(REPO_ID, "artifacts/class_names.json")
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    state_path = hf_download(REPO_ID, "artifacts/model_state.pt")

    # Build same architecture used in training
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))

    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, class_names

model, class_names = load_model_and_labels()

tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input image", use_container_width=True)

    x = tfms(img).unsqueeze(0)  # (1, 3, H, W)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy().reshape(-1)

    top_idx = int(np.argmax(probs))
    st.metric("Prediction", class_names[top_idx])
    st.write({class_names[i]: float(probs[i]) for i in range(len(class_names))})
else:
    st.info("Upload an image to run inference.")
