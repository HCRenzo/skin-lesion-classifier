#!/usr/bin/env python3
"""Versiona el modelo ya entrenado (model/model.pt) en el Model Registry de
Unity Catalog, usando MLflow con Databricks Free Edition como backend.

Qué hace:
  1. Recalcula métricas de validación (mismo split que el entrenamiento,
     ver skin_classifier.train_valid_split).
  2. Loguea un run en MLflow: params, métricas por clase, matriz de
     confusión como artefacto.
  3. Empaqueta el modelo como pyfunc (predicción + Grad-CAM, ver
     mlflow_model.py) y lo registra como nueva versión en Unity Catalog.
  4. Mueve el alias `champion` a esa nueva versión — es lo que
     scripts/deploy_endpoint.py usa como "la última versión" para
     promoverla al endpoint que llama la app.

Requiere DATABRICKS_HOST y DATABRICKS_TOKEN en el entorno (ver README →
Anexo: MLflow + Databricks).

Uso: ./env_skin/bin/python scripts/register_model.py
"""

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlflow_model import SkinLesionPyfuncModel
from skin_classifier import (
    CLASS_NAMES,
    METADATA_FILE,
    MODEL_DIR,
    SkinLesionDataset,
    build_model,
    train_valid_split,
)

UC_MODEL_NAME = os.environ.get("UC_MODEL_NAME", "main.default.skin_lesion_classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
EXPERIMENT_PATH = os.environ.get("MLFLOW_EXPERIMENT_PATH", "/Shared/skin-lesion-classifier")
BATCH_SIZE = 32

# Gate de calidad (patrón champion/challenger): no mover el alias `champion`
# a la versión nueva si es peor que la actual en esta métrica -- evita que
# un reentrenamiento con peor suerte (o un bug) pise silenciosamente al
# modelo que está sirviendo la app. FORCE_PROMOTE=1 lo saltea.
GATE_METRIC = os.environ.get("GATE_METRIC", "f1_macro")
FORCE_PROMOTE = os.environ.get("FORCE_PROMOTE", "").lower() in ("1", "true", "yes")


def evaluate(model, valid_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in valid_loader:
            preds = model(imgs).argmax(dim=1).numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    # labels=range(...) explícito: sin esto, si una muestra chica (smoke
    # test con HAM10000_MAX_SAMPLES) no incluye las 7 clases por azar,
    # tanto classification_report como confusion_matrix explotan o
    # devuelven una matriz de menor tamaño que CLASS_NAMES.
    all_class_ids = list(range(len(CLASS_NAMES)))
    report = classification_report(
        all_labels,
        all_preds,
        labels=all_class_ids,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=all_class_ids)
    return report, cm


def main():
    # Corriendo dentro de un cluster/job de Databricks, la autenticación es
    # implícita (no hace falta host/token) -- DATABRICKS_RUNTIME_VERSION
    # solo existe ahí. Fuera de Databricks (tu máquina, GitHub Actions),
    # sí hacen falta explícitos.
    running_in_databricks = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))
    if not running_in_databricks and (
        not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN")
    ):
        sys.exit(
            "Faltan DATABRICKS_HOST y/o DATABRICKS_TOKEN en el entorno.\n"
            "Ver README -> Anexo: MLflow + Databricks para cómo generarlos."
        )

    model_path = MODEL_DIR / "model.pt"
    if not model_path.exists():
        sys.exit(f"No existe {model_path} -- corré primero: ./env_skin/bin/python scripts/train_model.py")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT_PATH)

    df = pd.read_csv(METADATA_FILE)
    _, valid_df = train_valid_split(df, seed=42, train_frac=0.8)
    max_samples = os.environ.get("HAM10000_MAX_SAMPLES")
    if max_samples:
        valid_df = valid_df.sample(min(int(max_samples), len(valid_df)), random_state=42)
    valid_ds = SkinLesionDataset(valid_df, train=False)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    print("Evaluando checkpoint sobre el split de validación (lesion_id, seed=42)...")
    report, cm = evaluate(model, valid_loader)
    print(f"Accuracy: {report['accuracy']:.4f} | F1 macro: {report['macro avg']['f1-score']:.4f}")

    new_metrics = {"accuracy": report["accuracy"], "f1_macro": report["macro avg"]["f1-score"]}
    for cls in CLASS_NAMES:
        new_metrics[f"f1_{cls}"] = report[cls]["f1-score"]
        new_metrics[f"recall_{cls}"] = report[cls]["recall"]
        new_metrics[f"precision_{cls}"] = report[cls]["precision"]

    # Gate: comparar contra el champion actual (si existe) antes de decidir
    # si esta versión nueva merece quedar sirviendo en el endpoint.
    client = mlflow.MlflowClient()
    current_metric = None
    try:
        current_mv = client.get_model_version_by_alias(UC_MODEL_NAME, MODEL_ALIAS)
        current_run = client.get_run(current_mv.run_id)
        current_metric = current_run.data.metrics.get(GATE_METRIC)
        print(f"Champion actual: {UC_MODEL_NAME}@{MODEL_ALIAS} = v{current_mv.version} ({GATE_METRIC}={current_metric})")
    except Exception:
        print(f"No hay alias '{MODEL_ALIAS}' todavía -- esta va a ser la primera versión registrada.")

    should_promote = FORCE_PROMOTE or current_metric is None or new_metrics[GATE_METRIC] >= current_metric

    with mlflow.start_run(run_name="resnet18-ham10000") as run:
        mlflow.log_params(
            {
                "architecture": "resnet18",
                "pretrained": "imagenet",
                "num_classes": len(CLASS_NAMES),
                "split": "80/20 por lesion_id, seed=42",
            }
        )
        for name, value in new_metrics.items():
            mlflow.log_metric(name, value)

        cm_path = Path("confusion_matrix.csv")
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cm_path)
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink()

        # Firma del modelo: input = 1 columna con la imagen en base64,
        # output = lista de {probs, gradcam} (ver mlflow_model.py).
        sample_input = pd.DataFrame({"image_b64": ["<imagen PNG en base64>"]})
        sample_output = [{"probs": {c: 0.0 for c in CLASS_NAMES}, "gradcam": [[0.0] * 7] * 7}]
        signature = infer_signature(sample_input, sample_output)

        # conda_env explícito (en vez de dejar que mlflow infiera python=3.9.6
        # exacto del intérprete local): el canal conda privado de Databricks
        # Model Serving no siempre tiene ese build exacto de Python -- fijar
        # solo major.minor ("3.9") le da margen al solver para resolver.
        conda_env = {
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.9",
                "pip",
                {
                    "pip": [
                        "mlflow==3.1.4",
                        "torch==2.8.0",
                        "torchvision==0.23.0",
                        "pillow",
                        "pandas",
                        "numpy",
                        # skin_classifier.py (empaquetado vía code_paths) importa
                        # `requests` a nivel de módulo -- sin esto, load_context()
                        # falla en el servidor con "missing Python dependency".
                        "requests",
                    ]
                },
            ],
            "name": "mlflow-env",
        }

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SkinLesionPyfuncModel(),
            artifacts={"weights": str(model_path)},
            code_paths=[
                str(Path(__file__).resolve().parent / "skin_classifier.py"),
                str(Path(__file__).resolve().parent / "mlflow_model.py"),
            ],
            signature=signature,
            input_example=sample_input,
            conda_env=conda_env,
        )

        mv = mlflow.register_model(model_uri=model_info.model_uri, name=UC_MODEL_NAME)

        print(f"\nRun: {run.info.run_id}")
        print(f"Modelo registrado: {UC_MODEL_NAME} v{mv.version}")

        if should_promote:
            client.set_registered_model_alias(UC_MODEL_NAME, MODEL_ALIAS, mv.version)
            print(f"Alias '{MODEL_ALIAS}' -> v{mv.version} (promovido)")
            print("\nSiguiente paso: ./env_skin/bin/python scripts/deploy_endpoint.py")
        else:
            print(
                f"NO promovido: {GATE_METRIC}={new_metrics[GATE_METRIC]:.4f} es peor que el "
                f"champion actual ({current_metric:.4f}). Queda registrado como v{mv.version} "
                f"pero '{MODEL_ALIAS}' sigue en la versión anterior.\n"
                f"(Forzar con FORCE_PROMOTE=1 si igual querés promoverlo.)"
            )


if __name__ == "__main__":
    main()
