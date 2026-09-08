#!/usr/bin/env python3
"""Sube scripts/*.py al Workspace de Databricks como archivos planos
importables (no notebooks) -- es el paso de CD que mantiene actualizado el
código que corre el Job de reentrenamiento (ver ct_pipeline.py) cada vez
que cambia algo en scripts/.

Requiere DATABRICKS_HOST y DATABRICKS_TOKEN (localmente, o como secrets en
el workflow de GitHub Actions -- ver .github/workflows/retrain.yml).

Uso: ./env_skin/bin/python scripts/sync_to_databricks.py
"""

import os
import sys
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat

WORKSPACE_BASE_PATH = os.environ.get(
    "DATABRICKS_WORKSPACE_PATH", "/Workspace/Shared/skin_lesion_classifier/scripts"
)

FILES = [
    "skin_classifier.py",
    "mlflow_model.py",
    "train_model.py",
    "register_model.py",
    "deploy_endpoint.py",
    "ct_pipeline.py",
]


def main():
    if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
        sys.exit("Faltan DATABRICKS_HOST y/o DATABRICKS_TOKEN en el entorno.")

    scripts_dir = Path(__file__).resolve().parent
    w = WorkspaceClient()
    w.workspace.mkdirs(WORKSPACE_BASE_PATH)

    for name in FILES:
        local_path = scripts_dir / name
        remote_path = f"{WORKSPACE_BASE_PATH}/{name}"
        with open(local_path, "rb") as f:
            w.workspace.upload(remote_path, f, format=ImportFormat.AUTO, overwrite=True)
        print(f"  {name} -> {remote_path}")

    print(f"\n{len(FILES)} archivos sincronizados en {WORKSPACE_BASE_PATH}")


if __name__ == "__main__":
    main()
