#!/usr/bin/env python3
"""Genera las imágenes usadas en el readme: predicciones de ejemplo con
Grad-CAM, matriz de confusión, y distribución de clases del dataset.

Uso: ./env_skin/bin/python scripts/make_readme_images.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skin_classifier import (
    CLASS_LABELS_ES,
    CLASS_NAMES,
    METADATA_FILE,
    find_image_path,
    grad_cam,
    load_model,
    predict_image,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def overlay_heatmap(pil_image: Image.Image, cam: np.ndarray) -> Image.Image:
    base = pil_image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    base_arr = np.asarray(base).astype("float32")
    heat = np.zeros_like(base_arr)
    heat[..., 0] = cam * 255
    blended = (base_arr * 0.6 + heat * 0.4).clip(0, 255).astype("uint8")
    return Image.fromarray(blended)


def make_prediction_grid(df, model):
    """3 ejemplos: uno bien clasificado de la clase mayoritaria, uno de una
    clase rara bien clasificado, y un melanoma para mostrar la limitación
    conocida (recall bajo) con transparencia."""
    rng = np.random.default_rng(7)

    examples = []
    # Un nv (benigno) bien clasificado
    nv_rows = df[df["dx"] == "nv"]
    for _, row in nv_rows.sample(20, random_state=1).iterrows():
        path = find_image_path(row["image_id"])
        if not path:
            continue
        img = Image.open(path)
        probs = predict_image(model, img)
        pred = max(probs, key=probs.get)
        if pred == "nv":
            examples.append((img, row["dx"], pred, probs))
            break

    # Un bcc (carcinoma) bien clasificado
    bcc_rows = df[df["dx"] == "bcc"]
    for _, row in bcc_rows.sample(20, random_state=2).iterrows():
        path = find_image_path(row["image_id"])
        if not path:
            continue
        img = Image.open(path)
        probs = predict_image(model, img)
        pred = max(probs, key=probs.get)
        if pred == "bcc":
            examples.append((img, row["dx"], pred, probs))
            break

    # Un melanoma — mostrar cualquiera, acierte o no, para ser honestos
    mel_rows = df[df["dx"] == "mel"]
    for _, row in mel_rows.sample(20, random_state=3).iterrows():
        path = find_image_path(row["image_id"])
        if not path:
            continue
        img = Image.open(path)
        probs = predict_image(model, img)
        pred = max(probs, key=probs.get)
        examples.append((img, row["dx"], pred, probs))
        break

    fig, axes = plt.subplots(2, len(examples), figsize=(4 * len(examples), 8))
    for i, (img, true_dx, pred_dx, probs) in enumerate(examples):
        class_idx = CLASS_NAMES.index(pred_dx)
        cam = grad_cam(model, img, class_idx)
        overlay = overlay_heatmap(img, cam)

        axes[0, i].imshow(img.resize((224, 224)))
        axes[0, i].axis("off")
        axes[0, i].set_title("Imagen original", fontsize=10)

        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")
        acierto = "(acertó)" if true_dx == pred_dx else "(NO acertó)"
        axes[1, i].set_title(
            f"Grad-CAM\nReal: {CLASS_LABELS_ES[true_dx]}\n"
            f"Predijo: {CLASS_LABELS_ES[pred_dx]} ({probs[pred_dx]:.0%}) {acierto}",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "predicciones_ejemplo.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Guardado: predicciones_ejemplo.png")


def make_confusion_matrix(df, model, valid_lesions):
    valid_df = df[df["lesion_id"].isin(valid_lesions)]
    y_true, y_pred = [], []
    for _, row in valid_df.iterrows():
        path = find_image_path(row["image_id"])
        if not path:
            continue
        img = Image.open(path)
        probs = predict_image(model, img)
        y_true.append(row["dx"])
        y_pred.append(max(probs, key=probs.get))

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (normalizada por fila)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(
                j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9,
            )
    plt.colorbar(im, ax=ax, label="proporción dentro de la clase real")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "matriz_confusion.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Guardado: matriz_confusion.png")


def make_class_distribution(df):
    counts = df["dx"].value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([CLASS_LABELS_ES[c] for c in CLASS_NAMES], counts.values, color="#4C72B0")
    ax.set_ylabel("Cantidad de imágenes")
    ax.set_title("Distribución de clases en HAM10000 (10,015 imágenes)")
    plt.xticks(rotation=30, ha="right")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, str(count), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "distribucion_clases.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Guardado: distribucion_clases.png")


def main():
    df = pd.read_csv(METADATA_FILE)
    unique_lesions = df["lesion_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_lesions)
    n_train = int(len(unique_lesions) * 0.8)
    valid_lesions = set(unique_lesions[n_train:])

    device = "cpu"
    model = load_model(device=device)

    make_class_distribution(df)
    make_prediction_grid(df, model)
    make_confusion_matrix(df, model, valid_lesions)


if __name__ == "__main__":
    main()
