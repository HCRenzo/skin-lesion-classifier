#!/usr/bin/env python3
"""Entry point que corre DENTRO de un Job de Databricks: entrena, evalúa,
registra (con gate de calidad) y promueve el modelo -- todo leyendo el
dataset directo del Volume de Unity Catalog, sin depender de una laptop.

Esto es lo que hace "CT" (continuous training) real: un Job programado (o
disparado desde GitHub Actions) corre este script de punta a punta.

No se corre directo: scripts/sync_to_databricks.py sube este archivo (y el
resto de scripts/) al Workspace, y el notebook del Job (creado por
scripts/create_ct_job.py) lo importa y lo llama. Ver Anexo "CI/CD/CT" del
readme.
"""

import os

VOLUME_BASE = "/Volumes/main/default/ham10000_data"

# Mismos nombres de env var que lee scripts/skin_classifier.py -- acá se
# pisan para apuntar al Volume en vez de a data/ local. setdefault() para
# no pisar si alguien ya los seteó a mano (ej. para probar con un Volume
# distinto).
os.environ.setdefault("HAM10000_DATA_DIR", VOLUME_BASE)
os.environ.setdefault("HAM10000_METADATA_FILE", f"{VOLUME_BASE}/HAM10000_metadata.csv")
os.environ.setdefault(
    "HAM10000_IMAGE_DIRS",
    f"{VOLUME_BASE}/images/data/HAM10000_images_part_1:{VOLUME_BASE}/images/data/HAM10000_images_part_2",
)
os.environ.setdefault("HAM10000_MODEL_DIR", f"{VOLUME_BASE}/models")

import deploy_endpoint  # noqa: E402
import register_model  # noqa: E402
import train_model  # noqa: E402


def main():
    print("=== [CT 1/3] Entrenando ===")
    train_model.main()

    print("\n=== [CT 2/3] Evaluando y registrando (con gate de calidad) ===")
    register_model.main()

    print("\n=== [CT 3/3] Actualizando el endpoint de serving ===")
    deploy_endpoint.main()


if __name__ == "__main__":
    main()
