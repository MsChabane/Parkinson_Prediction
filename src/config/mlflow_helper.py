from dotenv import load_dotenv
import os
import yaml
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _active_run_stack
import dagshub
load_dotenv()


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def configure_mlflow(experiment_name: str):
    org = os.getenv("DAGSHUB_ORG_NAME", "")
    repo = os.getenv("DAGSHUB_REPO_NAME", "")
    user = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")

    if org and repo and token and token != "YOUR_PERSONAL_TOKEN":
        uri = f"https://dagshub.com/{org}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_URI"] = uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        dagshub.auth.add_app_token(token)
        mlflow.set_tracking_uri(uri)
    else:
        
        mlflow.set_tracking_uri("file://./artifacts/mlruns")

    mlflow.set_experiment(experiment_name)


def clear_active_runs():
    while mlflow.active_run() is not None:
        try:
            mlflow.end_run()
        except Exception:
            if len(_active_run_stack.get()) > 0:
                _active_run_stack.get().pop()


def get_or_create_shared_run(params: dict, create_new: bool = False):
    mlflow_cfg = params["mlflow"]
    configure_mlflow(mlflow_cfg["experiment_name"])
    clear_active_runs()

    run_id_file = Path(mlflow_cfg["run_id_file"])
    run_id_file.parent.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()

    if mlflow_cfg["reuse_shared_run_id"] and not create_new and run_id_file.exists():
        saved_id = run_id_file.read_text().strip()
        if saved_id:
            try:
                client.update_run(saved_id, status="RUNNING")
                return mlflow.start_run(run_id=saved_id)
            except Exception:
                pass

    run = mlflow.start_run(run_name=mlflow_cfg["run_name"])
    run_id_file.write_text(run.info.run_id)
    return run