#!/usr/bin/env python3
"""Crea (o actualiza) el endpoint de Databricks Model Serving para que
sirva la versión del modelo apuntada por el alias `champion` en Unity
Catalog -- es decir, la última versión registrada por register_model.py.

Correr esto después de cada register_model.py para "promover" la nueva
versión al endpoint que usa la app (app.py llama siempre al mismo nombre
de endpoint; lo que cambia es qué versión sirve por detrás).

Requiere DATABRICKS_HOST y DATABRICKS_TOKEN en el entorno (ver README ->
Anexo: MLflow + Databricks).

Nota: los nombres de método/parámetro del SDK de Databricks pueden variar
entre versiones. Si algo falla acá, es más rápido revisar la doc actual de
`databricks-sdk` (o crear/actualizar el endpoint a mano en Serving del
workspace) que depurar a ciegas -- no tengo forma de probar esto contra tu
workspace real.

Uso: ./env_skin/bin/python scripts/deploy_endpoint.py
"""

import os
import sys

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

UC_MODEL_NAME = os.environ.get("UC_MODEL_NAME", "main.default.skin_lesion_classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "skin-lesion-classifier")
# Free Edition: compute serverless CPU -- sin GPU. "Small" es el tamaño más
# chico disponible; ajustar si el workspace ofrece otras opciones.
WORKLOAD_SIZE = os.environ.get("SERVING_WORKLOAD_SIZE", "Small")


def main():
    # Dentro de un cluster/job de Databricks la autenticación es implícita
    # (ver misma nota en register_model.py).
    running_in_databricks = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))
    if not running_in_databricks and (
        not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN")
    ):
        sys.exit(
            "Faltan DATABRICKS_HOST y/o DATABRICKS_TOKEN en el entorno.\n"
            "Ver README -> Anexo: MLflow + Databricks."
        )

    mlflow.set_registry_uri("databricks-uc")
    mv = mlflow.MlflowClient().get_model_version_by_alias(UC_MODEL_NAME, MODEL_ALIAS)
    version = str(mv.version)
    print(f"Alias '{MODEL_ALIAS}' -> {UC_MODEL_NAME} v{version}")

    w = WorkspaceClient()
    served_entity = ServedEntityInput(
        entity_name=UC_MODEL_NAME,
        entity_version=version,
        workload_size=WORKLOAD_SIZE,
        scale_to_zero_enabled=True,
    )

    try:
        w.serving_endpoints.get(ENDPOINT_NAME)
        exists = True
    except ResourceDoesNotExist:
        exists = False

    if not exists:
        print(f"Creando endpoint '{ENDPOINT_NAME}' (puede tardar varios minutos)...")
        w.serving_endpoints.create_and_wait(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),
        )
    else:
        print(f"Actualizando endpoint '{ENDPOINT_NAME}' a la v{version} (puede tardar varios minutos)...")
        w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME,
            served_entities=[served_entity],
        )

    host = w.config.host.rstrip("/")
    print("\nListo. La app debe apuntar a:")
    print(f"  SERVING_ENDPOINT_NAME={ENDPOINT_NAME}")
    print(f"  URL de invocación: {host}/serving-endpoints/{ENDPOINT_NAME}/invocations")


if __name__ == "__main__":
    main()
