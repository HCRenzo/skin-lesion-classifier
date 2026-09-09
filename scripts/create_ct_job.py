#!/usr/bin/env python3
"""Crea (o actualiza) el Job de Databricks que hace CT: entrena, evalúa,
registra con gate de calidad, y actualiza el endpoint -- todo corriendo
adentro de Databricks, leyendo el dataset del Volume. Programado semanal
por default (CronSchedule) además de poder dispararse a mano.

Correr esto UNA VEZ para crear el Job (o de nuevo si cambia el schedule o
la ubicación del notebook). El código que ejecuta el Job se actualiza
aparte con scripts/sync_to_databricks.py -- correr eso cada vez que cambie
algo en scripts/, no hace falta volver a correr este.

Requiere DATABRICKS_HOST y DATABRICKS_TOKEN.

Uso: ./env_skin/bin/python scripts/create_ct_job.py
"""

import base64
import os
import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import Environment
from databricks.sdk.service.jobs import (
    CronSchedule,
    JobEnvironment,
    JobSettings,
    NotebookTask,
    PauseStatus,
    Task,
)
from databricks.sdk.service.workspace import ImportFormat, Language

# El notebook serverless por default no trae torch/mlflow -- hay que
# declarar el entorno explícito. Mismas versiones que requirements.txt y
# que el conda_env de register_model.py (tienen que coincidir con lo que
# ya corre localmente/en el endpoint).
ENV_DEPENDENCIES = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "mlflow==3.1.4",
    "databricks-sdk==0.102.0",
    "scikit-learn==1.6.1",
    "pillow==11.3.0",
    "requests==2.32.5",
    "pandas==2.3.3",
    "numpy==2.0.2",
]

JOB_NAME = os.environ.get("CT_JOB_NAME", "skin-lesion-classifier-ct")
WORKSPACE_BASE = os.environ.get("DATABRICKS_WORKSPACE_PATH", "/Workspace/Shared/skin_lesion_classifier/scripts")
ENTRYPOINT_NOTEBOOK_PATH = "/Workspace/Shared/skin_lesion_classifier/ct_entrypoint"
CRON_SCHEDULE = os.environ.get("CT_CRON_SCHEDULE", "0 0 3 ? * SUN")  # domingos 03:00 UTC
CRON_TIMEZONE = os.environ.get("CT_CRON_TIMEZONE", "UTC")

ENTRYPOINT_NOTEBOOK_SOURCE = f"""# Databricks notebook source
import sys
sys.path.insert(0, "{WORKSPACE_BASE}")

import os
dbutils.widgets.text("epochs", "")
dbutils.widgets.text("max_samples", "")
epochs = dbutils.widgets.get("epochs")
max_samples = dbutils.widgets.get("max_samples")
if epochs:
    os.environ["HAM10000_EPOCHS"] = epochs
if max_samples:
    os.environ["HAM10000_MAX_SAMPLES"] = max_samples

import ct_pipeline
ct_pipeline.main()
"""


def main():
    if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
        sys.exit("Faltan DATABRICKS_HOST y/o DATABRICKS_TOKEN en el entorno.")

    w = WorkspaceClient()

    w.workspace.import_(
        path=ENTRYPOINT_NOTEBOOK_PATH,
        content=base64.b64encode(ENTRYPOINT_NOTEBOOK_SOURCE.encode()).decode(),
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        overwrite=True,
    )
    print(f"Notebook de entrada: {ENTRYPOINT_NOTEBOOK_PATH}")

    environment_key = "ct-env"
    environments = [JobEnvironment(environment_key=environment_key, spec=Environment(
        environment_version="2", dependencies=ENV_DEPENDENCIES
    ))]

    task = Task(
        task_key="train_register_deploy",
        notebook_task=NotebookTask(notebook_path=ENTRYPOINT_NOTEBOOK_PATH),
        environment_key=environment_key,
        timeout_seconds=5 * 60 * 60,  # 5h -- fine-tuning de ResNet18 en CPU serverless
    )
    schedule = CronSchedule(
        quartz_cron_expression=CRON_SCHEDULE, timezone_id=CRON_TIMEZONE, pause_status=PauseStatus.UNPAUSED
    )

    existing = next((j for j in w.jobs.list(name=JOB_NAME)), None)

    settings = JobSettings(name=JOB_NAME, tasks=[task], schedule=schedule, environments=environments)

    if existing is None:
        resp = w.jobs.create(name=JOB_NAME, tasks=[task], schedule=schedule, environments=environments)
        job_id = resp.job_id
        print(f"Job creado: {JOB_NAME} (id={job_id})")
    else:
        job_id = existing.job_id
        w.jobs.reset(job_id=job_id, new_settings=settings)
        print(f"Job actualizado: {JOB_NAME} (id={job_id})")

    print(f"Schedule: '{CRON_SCHEDULE}' ({CRON_TIMEZONE})")
    print("\nPara disparar un run manual: ./env_skin/bin/python scripts/trigger_ct_job.py")


if __name__ == "__main__":
    main()
