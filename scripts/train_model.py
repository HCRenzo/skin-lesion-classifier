#!/usr/bin/env python3
"""Entrena el clasificador de lesiones de piel (HAM10000): ResNet18
pre-entrenado en ImageNet, fine-tuneado para las 7 clases del dataset.

Split train/valid por lesion_id (no por imagen) — algunas lesiones tienen
más de una foto, y mezclarlas entre train/valid sería fuga de datos, el
mismo error que ya evitamos en el proyecto de Dota con match_id.

Loss ponderada por clase para compensar el desbalance (nv es 67% del
dataset, df es 1.1%).

Uso: ./env_skin/bin/python scripts/train_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skin_classifier import CLASS_NAMES, METADATA_FILE, MODEL_DIR, SkinLesionDataset, build_model

EPOCHS = 15
PATIENCE = 4
BATCH_SIZE = 32
NUM_WORKERS = 2


def main():
    df = pd.read_csv(METADATA_FILE)

    unique_lesions = df["lesion_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_lesions)
    n_train = int(len(unique_lesions) * 0.8)
    train_lesions = set(unique_lesions[:n_train])
    valid_lesions = set(unique_lesions[n_train:])

    train_df = df[df["lesion_id"].isin(train_lesions)]
    valid_df = df[df["lesion_id"].isin(valid_lesions)]
    print(f"Lesiones train: {len(train_lesions)} | valid: {len(valid_lesions)}")
    print(f"Imágenes train: {len(train_df)} | valid: {len(valid_df)}")

    train_ds = SkinLesionDataset(train_df, train=True)
    valid_ds = SkinLesionDataset(valid_df, train=False)

    class_counts = train_df["dx"].value_counts()
    class_weights = torch.tensor(
        [1.0 / class_counts.get(c, 1) for c in CLASS_NAMES], dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)
    print("Pesos por clase (mayor peso = clase más rara):")
    for c, w in zip(CLASS_NAMES, class_weights.tolist()):
        print(f"  {c}: {w:.2f}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_valid_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)

        model.eval()
        valid_loss = 0.0
        correct = 0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                valid_loss += loss.item() * len(imgs)
                correct += (logits.argmax(dim=1) == labels).sum().item()
        valid_loss /= len(valid_ds)
        valid_acc = correct / len(valid_ds)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - train_loss: {train_loss:.4f} "
            f"- valid_loss: {valid_loss:.4f} - valid_acc: {valid_acc:.4f}"
        )

        if valid_loss < best_valid_loss - 1e-4:
            best_valid_loss = valid_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping en epoch {epoch + 1} (sin mejora en {PATIENCE} epochs).")
                break

    model.load_state_dict(best_state)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    print(f"\nMejor valid_loss: {best_valid_loss:.4f}")
    print(f"Modelo guardado en: {MODEL_DIR}")

    # Reporte final: accuracy global no alcanza con este desbalance,
    # hace falta precision/recall/F1 por clase.
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n=== Classification report (validación) ===")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print("=== Matriz de confusión ===")
    print("Filas = real, columnas = predicho. Orden:", CLASS_NAMES)
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
