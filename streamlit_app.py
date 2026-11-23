"""
streamlit_app.py - Retinal OCT classifier (educational prototype).

Setup:  put best_model.pt (saved by the notebook) in the SAME folder as this file.
Run  :  streamlit run streamlit_app.py
"""
import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """A small CNN built from scratch.
    Each block = Convolution -> BatchNorm -> ReLU -> MaxPool
      Convolution : finds patterns (edges, retinal layers, fluid pockets)
      BatchNorm   : keeps numbers stable so training is faster
      ReLU        : non-linearity, lets the network learn curved decision rules
      MaxPool     : shrinks the image, keeping the strongest signals
    """

    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),             
            nn.Linear(128, num_classes),      
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name, num_classes=4, freeze_backbone=False, pretrained=True):
    """Create any of the 5 models.
    freeze_backbone=True  -> feature extraction (only the new last layer learns)
    freeze_backbone=False -> fine-tuning (everything learns)
    """
    if name == "custom_cnn":
        return CustomCNN(num_classes)

    weights = "DEFAULT" if pretrained else None
    if name == "resnet18":
        model = models.resnet18(weights=weights)
    elif name == "vgg16":
        model = models.vgg16(weights=weights)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
    else:
        raise ValueError("Unknown model: " + name)

    if freeze_backbone:                       
        for p in model.parameters():
            p.requires_grad = False

    
    if name == "resnet18":
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif name == "vgg16":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:   
        in_f = model.classifier[1].in_features if name == "mobilenet_v2" else model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    return model


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm


def get_target_layer(model, model_name):
    """The last convolutional stage of each architecture."""
    if model_name == "custom_cnn":
        return model.features[-1]
    if model_name == "resnet18":
        return model.layer4
    if model_name == "vgg16":
        return model.features[29]          
    if model_name == "mobilenet_v2":
        return model.features[-1]
    if model_name == "densenet121":
        return model.features
    raise ValueError("No Grad-CAM layer for " + model_name)


class GradCAM:
    """Grad-CAM: shows WHERE the model looked when it made its decision.
    1) keep the feature maps of the last conv stage  2) backpropagate from the predicted score
    3) average the gradients = importance of each map  4) weighted sum + ReLU = heat-map
    """

    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        out = output.clone()               
        self.activations = out.detach().clone()
        out.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))
        return out

    def remove(self):
        self.handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        """input_tensor: (1, 3, H, W). Returns (heat-map HxW in [0,1], class_idx)."""
        x = input_tensor.clone().requires_grad_(True)   
        with torch.enable_grad():
            scores = self.model(x)
            if class_idx is None:
                class_idx = int(scores.argmax(1))
            self.model.zero_grad()
            scores[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx


def overlay_heatmap(image_rgb, heatmap, alpha=0.45):
    """Blend the heat-map on the picture. image_rgb: HxWx3 in [0,1]. Blue = low, red = high attention."""
    colored = cm.jet(heatmap)[..., :3]
    return np.clip((1 - alpha) * image_rgb + alpha * colored, 0, 1)




MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
IMAGE_SIZE = 224
MAX_FILE_MB = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def denormalize(tensor):
    """Undo normalization so the tensor can be displayed as a picture."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


@st.cache_resource            
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = build_model(ckpt["model_name"], pretrained=False)   
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["class_names"], ckpt["model_name"]



st.set_page_config(page_title="Retinal OCT Classifier", page_icon="👁️", layout="wide")
st.title("👁️ Retinal Disease Detection from OCT Images")
st.warning("**Disclaimer:** This application is an educational prototype and is **not** a "
           "clinical diagnostic tool. Do not use it for medical decisions.")

loaded = load_model()
if loaded is None:
    st.error("`best_model.pt` was not found next to this file. Run the notebook, download "
             "`best_model.pt`, and place it in the same folder as `streamlit_app.py`.")
    st.stop()
model, class_names, model_name = loaded
st.caption(f"Loaded model: **{model_name}**  |  Classes: {', '.join(class_names)}")

uploaded = st.file_uploader("Upload a retinal OCT image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an OCT image to get started.")
    st.stop()

# ---- Input validation ----
if uploaded.size > MAX_FILE_MB * 1024 * 1024:
    st.error(f"File is larger than {MAX_FILE_MB} MB. Please upload a smaller image.")
    st.stop()
try:
    image = Image.open(uploaded).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
    st.stop()
if min(image.size) < 64:
    st.error("The image is too small to analyse (minimum 64 pixels per side).")
    st.stop()

# ---- Prediction ----
tensor = eval_tf(image).unsqueeze(0)                 
with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)[0].numpy()
pred_idx = int(np.argmax(probs))
confidence = float(probs[pred_idx]) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Uploaded image")
    st.image(image, width="stretch")
with col2:
    st.subheader("Prediction")
    st.metric("Predicted class", class_names[pred_idx])
    st.metric("Confidence", f"{confidence:.1f}%")
    if confidence < 60:
        st.info("Confidence is low. The image may be unusual or may not be an OCT scan.")
    st.bar_chart({c: float(p) for c, p in zip(class_names, probs)})
with col3:
    st.subheader("Grad-CAM")
    try:
        cam_tool = GradCAM(model, get_target_layer(model, model_name))
        heatmap, _ = cam_tool(tensor, pred_idx)
        cam_tool.remove()
        base = denormalize(tensor[0]).permute(1, 2, 0).numpy()
        st.image(overlay_heatmap(base, heatmap), width="stretch",
                 caption="Red areas influenced the prediction most")
    except Exception as err:                           
        st.info(f"Grad-CAM could not be generated: {err}")
