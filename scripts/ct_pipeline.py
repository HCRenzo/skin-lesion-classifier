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
import shutil
import time
from pathlib import Path

VOLUME_BASE = "/Volumes/main/default/ham10000_data"
# El Volume es almacenamiento por red (rápido para un archivo grande, lento
# para miles de reads chicos y random como hace un DataLoader en cada
# época). Copiar una vez a disco local del compute y entrenar desde ahí
# fue lo que hizo falta para que un entrenamiento completo (15 épocas)
# entre en el timeout del Job -- ver Anexo "CI/CD/CT" del readme.
LOCAL_IMAGE_CACHE = Path("/tmp/ham10000_images_cache")
IMAGE_PART_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]


def _cache_images_locally() -> str:
    volume_root = Path(VOLUME_BASE) / "images" / "data"
    local_paths = [LOCAL_IMAGE_CACHE / d for d in IMAGE_PART_DIRS]

    if all(p.exists() for p in local_paths):
        print(f"Cache local ya presente en {LOCAL_IMAGE_CACHE}, no vuelvo a copiar.")
    else:
        LOCAL_IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for name, dst in zip(IMAGE_PART_DIRS, local_paths):
            src = volume_root / name
            if dst.exists():
                continue
            print(f"Copiando {src} -> {dst} ...")
            shutil.copytree(src, dst)
        print(f"Copia local del dataset lista en {time.time() - t0:.1f}s")

    return ":".join(str(p) for p in local_paths)


# Mismos nombres de env var que lee scripts/skin_classifier.py -- acá se
# pisan para apuntar al Volume/cache local en vez de a data/ local.
# setdefault() para no pisar si alguien ya los seteó a mano (ej. para
# probar con un Volume distinto).
os.environ.setdefault("HAM10000_DATA_DIR", VOLUME_BASE)
os.environ.setdefault("HAM10000_METADATA_FILE", f"{VOLUME_BASE}/HAM10000_metadata.csv")
os.environ.setdefault("HAM10000_IMAGE_DIRS", _cache_images_locally())
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
