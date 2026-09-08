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

import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skin_classifier import (
    CLASS_NAMES,
    METADATA_FILE,
    MODEL_DIR,
    SkinLesionDataset,
    build_model,
    train_valid_split,
)

# Overrideables por env var -- usado para smoke tests del pipeline de CT
# (ej. HAM10000_EPOCHS=1) sin tener que esperar un entrenamiento completo.
# En producción, dejar los defaults.
EPOCHS = int(os.environ.get("HAM10000_EPOCHS", 15))
PATIENCE = int(os.environ.get("HAM10000_PATIENCE", 4))
BATCH_SIZE = 32
# Los workers del DataLoader usan multiprocessing con memoria compartida
# (/dev/shm) -- el contenedor serverless de un Job de Databricks la tiene
# muy restringida y los workers mueren con "Bus error" si NUM_WORKERS > 0.
# Local (Mac/tu máquina) sí tiene memoria compartida normal, por eso el
# default sigue siendo 2 ahí.
_default_workers = 0 if os.environ.get("DATABRICKS_RUNTIME_VERSION") else 2
NUM_WORKERS = int(os.environ.get("HAM10000_NUM_WORKERS", _default_workers))

# Recortar el dataset a N imágenes (por split) -- solo para smoke tests
# rápidos del pipeline completo (CT) sin esperar leer/entrenar sobre el
# dataset entero desde el Volume. Vacío/no seteada = dataset completo.
MAX_SAMPLES = os.environ.get("HAM10000_MAX_SAMPLES")


def main():
    df = pd.read_csv(METADATA_FILE)
    train_df, valid_df = train_valid_split(df, seed=42, train_frac=0.8)
    if MAX_SAMPLES:
        n = int(MAX_SAMPLES)
        train_df = train_df.sample(min(n, len(train_df)), random_state=42)
        valid_df = valid_df.sample(min(n, len(valid_df)), random_state=42)
        print(f"HAM10000_MAX_SAMPLES={n} -- recortando dataset (smoke test, no usar para producción)")
    print(f"Lesiones train: {train_df['lesion_id'].nunique()} | valid: {valid_df['lesion_id'].nunique()}")
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

    # labels=range(...) explícito: con HAM10000_MAX_SAMPLES chico (smoke
    # tests) una muestra puede no incluir las 7 clases por azar, y sin esto
    # classification_report/confusion_matrix explotan o devuelven menos filas.
    all_class_ids = list(range(len(CLASS_NAMES)))
    print("\n=== Classification report (validación) ===")
    print(
        classification_report(
            all_labels, all_preds, labels=all_class_ids, target_names=CLASS_NAMES, zero_division=0
        )
    )
    print("=== Matriz de confusión ===")
    print("Filas = real, columnas = predicho. Orden:", CLASS_NAMES)
    print(confusion_matrix(all_labels, all_preds, labels=all_class_ids))


if __name__ == "__main__":
    main()
