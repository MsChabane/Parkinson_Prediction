import argparse
import shutil
from pathlib import Path
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from src.config.mlflow_helper import load_params, configure_mlflow, get_or_create_shared_run
from src.utils.logger import get_logger

logger = get_logger("download_data")

def resolve_source_run_id(run_id_arg: str, exp_tag: str, experiment_name: str) -> str:
    if run_id_arg and run_id_arg.upper() != "AUTO":
        return run_id_arg

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.error(f"Experiment '{experiment_name}' not found on MLflow server.")
        raise ValueError(f"Experiment '{experiment_name}' not found on MLflow server.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"params.exp_tag = '{exp_tag}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        logger.error(f"No run found with params.exp_tag = '{exp_tag}'. Pass --run-id explicitly.")
        raise ValueError(f"No run found with params.exp_tag = '{exp_tag}'. Pass --run-id explicitly.")
    return runs[0].info.run_id


def main():
    
    params = load_params()
    dl_cfg = params["download"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=dl_cfg["source_run_id"])
    parser.add_argument("--exp-tag", type=str, default=dl_cfg["exp_tag"])
    parser.add_argument("--embedding-path", type=str, default=dl_cfg["embedding_artifact_path"])
    parser.add_argument("--image-model-path", type=str, default=dl_cfg["image_model_artifact_path"])
    parser.add_argument("--voice-url", type=str, default=dl_cfg["voice_data_url"])
    parser.add_argument("--output-dir", type=str, default=dl_cfg["raw_dir"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_mlflow(params["mlflow"]["experiment_name"])

    target_emb_file = out_dir / "spiral_embeddings.pkl"
    target_img_model_file = out_dir / "spiral_image_model_with_schema.pkl"
    target_voice_file = out_dir / "voice_parkinsons.csv"



    try:
        source_run_id = resolve_source_run_id(args.run_id, args.exp_tag, params["mlflow"]["experiment_name"])
        emb_uri = f"runs:/{source_run_id}/{args.embedding_path}"
        mod_uri = f"runs:/{source_run_id}/{args.image_model_path}"

        dl_emb = mlflow.artifacts.download_artifacts(artifact_uri=emb_uri, dst_path=str(out_dir / "_tmp"))
        dl_mod = mlflow.artifacts.download_artifacts(artifact_uri=mod_uri, dst_path=str(out_dir / "_tmp"))

        shutil.move(dl_emb, target_emb_file)
        shutil.move(dl_mod, target_img_model_file)
        shutil.rmtree(out_dir / "_tmp", ignore_errors=True)
    except Exception as e:
         logger.error(f"Failed to download artifacts from MLflow: {e}")
         raise RuntimeError(f"Failed to download artifacts from MLflow: {e}")

    voice_df = pd.read_csv(args.voice_url)
    voice_df.to_csv(target_voice_file, index=False)

    with get_or_create_shared_run(params, create_new=True) as run:
        mlflow.log_params({
            "source_image_run_id": source_run_id,
            "source_exp_tag": args.exp_tag,
            "voice_dataset_rows": len(voice_df),
        })
        logger.info(f"Stage 1 Complete | Shared Run ID: {run.info.run_id}")
        logger.info(f"Saved: {target_emb_file}, {target_img_model_file}, {target_voice_file}")


if __name__ == "__main__":
    main()