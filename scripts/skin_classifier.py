#!/usr/bin/env python3
"""Lógica compartida: dataset, arquitectura (CNN con transfer learning) y
carga de modelo. Usado por el script de entrenamiento y la web."""

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
METADATA_FILE = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = [DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2"]

# Orden fijo y alfabético — tiene que ser el mismo en train y en inferencia.
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_LABELS_ES = {
    "akiec": "Queratosis actínica / carcinoma intraepitelial",
    "bcc": "Carcinoma basocelular",
    "bkl": "Lesión benigna tipo queratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus melanocítico (lunar benigno)",
    "vasc": "Lesión vascular",
}

# Clases que ameritan derivar a un dermatólogo (malignas o pre-malignas)
CONCERNING_CLASSES = {"akiec", "bcc", "mel"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def find_image_path(image_id: str):
    for d in IMAGE_DIRS:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def train_valid_split(df: pd.DataFrame, seed: int = 42, train_frac: float = 0.8):
    """Split 80/20 por `lesion_id` (no por imagen) — algunas lesiones tienen
    más de una foto, y mezclarlas entre train/valid sería fuga de datos.
    Compartido entre el entrenamiento (train_model.py) y el registro del
    modelo en MLflow (register_model.py) para que ambos evalúen sobre
    exactamente el mismo split."""
    unique_lesions = df["lesion_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_lesions)
    n_train = int(len(unique_lesions) * train_frac)
    train_lesions = set(unique_lesions[:n_train])
    valid_lesions = set(unique_lesions[n_train:])
    train_df = df[df["lesion_id"].isin(train_lesions)]
    valid_df = df[df["lesion_id"].isin(valid_lesions)]
    return train_df, valid_df


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SkinLesionDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.transform = get_transforms(train)
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = find_image_path(row["image_id"])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = self.class_to_idx[row["dx"]]
        return img, label


def build_model(num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> nn.Module:
    """ResNet18 pre-entrenado en ImageNet, con la última capa reemplazada
    para clasificar las 7 clases de HAM10000 (transfer learning)."""
    weights = "DEFAULT" if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(device: str = "cpu") -> nn.Module:
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model: nn.Module, pil_image: Image.Image, device: str = "cpu") -> dict:
    """Devuelve {clase: probabilidad} para las 7 clases."""
    transform = get_transforms(train=False)
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def grad_cam(model: nn.Module, pil_image: Image.Image, class_idx: int, device: str = "cpu") -> np.ndarray:
    """Mapa de calor Grad-CAM sobre la última capa convolucional (layer4 de
    ResNet18): muestra en qué región de la imagen se basó el modelo para
    predecir `class_idx`. Devuelve un array (IMAGE_SIZE, IMAGE_SIZE) en [0,1]."""
    transform = get_transforms(train=False)
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    target_layer = model.layer4[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        logits = model(x)
        score = logits[0, class_idx]
        score.backward()

        acts = activations["value"][0]  # (C, H, W)
        grads = gradients["value"][0]  # (C, H, W)
        weights = grads.mean(dim=(1, 2))  # (C,)
        cam = F.relu((weights[:, None, None] * acts).sum(dim=0))  # (H, W)
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(
            cam[None, None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0, 0]
        return cam.detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def _image_to_base64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_via_endpoint(
    pil_image: Image.Image,
    host: str,
    token: str,
    endpoint_name: str,
    timeout: float = 30.0,
) -> tuple[dict, np.ndarray]:
    """Llama al endpoint de Databricks Model Serving (última versión
    promovida vía el alias `champion` en Unity Catalog) y devuelve
    (probs, cam) — misma forma que predict_image()/grad_cam() locales, pero
    calculado del lado del servidor por el modelo pyfunc (ver
    scripts/mlflow_model.py). La app no necesita el peso del modelo local.

    El servidor manda el Grad-CAM en la resolución nativa de layer4 (7x7
    para input 224x224) para no inflar el payload — acá se reescala a
    IMAGE_SIZE x IMAGE_SIZE para superponerlo sobre la imagen original.
    """
    url = f"{host.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"
    payload = {
        "dataframe_split": {
            "columns": ["image_b64"],
            "data": [[_image_to_base64(pil_image)]],
        }
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    prediction = resp.json()["predictions"][0]

    probs = prediction["probs"]
    cam_small = torch.tensor(prediction["gradcam"], dtype=torch.float32)[None, None]
    cam = F.interpolate(
        cam_small, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )[0, 0]
    return probs, cam.numpy()
