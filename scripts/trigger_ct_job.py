#!/usr/bin/env python3
"""Dispara un run del Job de CT (creado por scripts/create_ct_job.py) a
demanda. Lo usa tanto vos a mano como el workflow de GitHub Actions
(.github/workflows/retrain.yml).

Uso:
  ./env_skin/bin/python scripts/trigger_ct_job.py                    # run completo
  HAM10000_EPOCHS=1 ./env_skin/bin/python scripts/trigger_ct_job.py  # smoke test rápido (1 epoch)
  CT_WAIT=0 ./env_skin/bin/python scripts/trigger_ct_job.py          # no esperar a que termine
"""

import os
import sys

from databricks.sdk import WorkspaceClient

JOB_NAME = os.environ.get("CT_JOB_NAME", "skin-lesion-classifier-ct")


def main():
    if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
        sys.exit("Faltan DATABRICKS_HOST y/o DATABRICKS_TOKEN en el entorno.")

    w = WorkspaceClient()
    job = next(iter(w.jobs.list(name=JOB_NAME)), None)
    if job is None:
        sys.exit(f"No existe el job '{JOB_NAME}' -- corré primero scripts/create_ct_job.py")

    notebook_params = {}
    epochs = os.environ.get("HAM10000_EPOCHS")
    if epochs:
        notebook_params["epochs"] = epochs
    max_samples = os.environ.get("HAM10000_MAX_SAMPLES")
    if max_samples:
        notebook_params["max_samples"] = max_samples

    print(
        f"Disparando run de '{JOB_NAME}' (id={job.job_id})"
        + (f" con epochs={epochs}" if epochs else " (config default)")
        + "..."
    )
    waiter = w.jobs.run_now(job_id=job.job_id, notebook_params=notebook_params)
    print(f"Run disparado: run_id={waiter.run_id}")
    print(f"Ver progreso: {w.config.host}/jobs/{job.job_id}/runs/{waiter.run_id}")

    if os.environ.get("CT_WAIT", "1") in ("0", "false", "False"):
        print("CT_WAIT=0 -- no espero a que termine.")
        return

    print("Esperando a que termine (puede tardar si entrena la config completa)...")
    run = waiter.result()
    result_state = run.state.result_state
    print(f"Resultado: {result_state}")
    if result_state is not None and "SUCCESS" not in str(result_state):
        sys.exit(1)


if __name__ == "__main__":
    main()
