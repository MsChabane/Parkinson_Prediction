import os
import shutil
import pickle
import argparse
import yaml
from mlflow.tracking import MlflowClient
from src.config.mlflow_helper import configure_mlflow
from src.utils.logger import get_logger

logger = get_logger("DownloadBestModel")


def find_best_multimodal_run(client: MlflowClient, cfg: dict) -> str:
    experiment_name = cfg["mlflow"]["experiment_name"]
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found on MLflow server.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    if not runs:
        raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")

    metric_candidates = [
        "best_test_f1",
        "best_multimodal_test_f1",
        "final_best_test_f1",
        "test_f1",
    ]

    best_run = None
    best_score = -1.0
    matched_metric = None

    for run in runs:
        for m_key in metric_candidates:
            if m_key in run.data.metrics:
                score = float(run.data.metrics[m_key])
                if score > best_score:
                    best_score = score
                    best_run = run
                    matched_metric = m_key
                break

    if best_run is not None:
        logger.info(
            f"Selected Best MLflow Run: {best_run.info.run_id} "
            f"({matched_metric} = {best_score:.4f})"
        )
        return best_run.info.run_id

    run_id_file = cfg["mlflow"].get("run_id_file", "artifacts/mlflow_run_id.txt")
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            saved_id = f.read().strip()
        if saved_id:
            logger.info(f"Using shared Run ID from {run_id_file}: {saved_id}")
            return saved_id

    fallback_id = runs[0].info.run_id
    logger.warning(f"No F1 metric found; falling back to latest run: {fallback_id}")
    return fallback_id


def download_artifact_to_path(
    client: MlflowClient, run_id: str, remote_path: str, local_dest: str
) -> str:
    os.makedirs(os.path.dirname(local_dest), exist_ok=True)
    tmp_dir = "/tmp/mlflow_best_model_dl"
    os.makedirs(tmp_dir, exist_ok=True)

    downloaded = client.download_artifacts(
        run_id=run_id, path=remote_path, dst_path=tmp_dir
    )
    if os.path.isdir(downloaded):
        for root, _, files in os.walk(downloaded):
            for file in files:
                if file.endswith(".pkl"):
                    downloaded = os.path.join(root, file)
                    break

    shutil.copy2(downloaded, local_dest)
    logger.info(f"Downloaded '{remote_path}' (run={run_id}) -> '{local_dest}'")
    return local_dest


def download_best_model_for_prediction(
    config_path: str = "params.yaml",
    run_id: str = None,
    download_dependencies: bool = True,
) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    experiment_name = cfg["mlflow"]["experiment_name"]
    tracking_uri = configure_mlflow(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    target_run_id = run_id or find_best_multimodal_run(client, cfg)

    best_model_local = "artifacts/models/multimodal_best_with_schema.pkl"
    candidate_remote_path = "models/multimodal_best/multimodal_best_with_schema.pkl"

    model_downloaded = False
    
    try:
        download_artifact_to_path(
            client=client,
            run_id=target_run_id,
            remote_path=candidate_remote_path,
            local_dest=best_model_local,
        )
        model_downloaded = True
        
    except Exception:
        logger.error(
            f"Failed to download best multimodal model from MLflow run {target_run_id}."
        )
        logger.info(
            "Ensure that the run contains 'multimodal_best_with_schema.pkl' artifact."
        )

    if not model_downloaded:
        raise FileNotFoundError(
            f"Could not find 'multimodal_best_with_schema.pkl' in MLflow run {target_run_id}."
        )

    if download_dependencies:
        preprocessor_files = [
            "voice_scaler.pkl",
            "img_emb_scaler.pkl",
            "img_pca.pkl",
            "unimodal_voice_models.pkl",
        ]
        for fname in preprocessor_files:
            try:
                download_artifact_to_path(
                    client=client,
                    run_id=target_run_id,
                    remote_path=f"preprocessors/{fname}",
                    local_dest=os.path.join("artifacts/preprocessors", fname),
                )
            except Exception as e:
                logger.warning(f"Preprocessor '{fname}' skipped: {e}")

        dl_cfg = cfg.get("download", {})
        source_img_run_id = dl_cfg.get("source_run_id")
        img_artifact_path = dl_cfg.get("image_model_artifact_path")
        raw_dir = dl_cfg.get("raw_dir", "data/raw")
        local_img_model = os.path.join(raw_dir, "spiral_image_model_with_schema.pkl")

        if source_img_run_id and img_artifact_path and not os.path.exists(local_img_model):
            try:
                download_artifact_to_path(
                    client=client,
                    run_id=source_img_run_id,
                    remote_path=img_artifact_path,
                    local_dest=local_img_model,
                )
            except Exception as e:
                logger.warning(f"Unimodal spiral image model download skipped: {e}")

    with open(best_model_local, "rb") as f:
        bundle = pickle.load(f)

    test_f1 = (
        bundle.get("metrics", {}).get("test", {}).get("f1")
        or bundle.get("test_metrics", {}).get("f1")
        or 0.0
    )
    logger.info(
        f"Successfully loaded best multimodal model: "
        f"name='{bundle.get('model_name')}', "
        f"fusion='{bundle.get('fusion_type')}', "
        f"test_f1={float(test_f1):.4f}"
    )
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the best Multimodal Parkinson's model from MLflow"
    )
    parser.add_argument("--config", type=str, default="params.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()

    download_best_model_for_prediction(
        config_path=args.config,
        run_id=args.run_id,
        download_dependencies=not args.model_only,
    )