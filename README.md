# Multimodal Early Detection of Parkinson's Disease

An end-to-end MLOps pipeline for early detection of Parkinson's Disease (PD) combining **Spiral Handwriting Drawings** (Computer Vision) and **Vocal Acoustic Biomarkers** (Signal/Tabular Analysis), orchestrated with **DVC**, tracked via **MLflow**, and hosted on **DagsHub**.

---

## Project Architecture & Highlights

- **Modality 1 — Spiral Drawing Vision Model (PyTorch)**:
  - Fine-tuned `mobilenet_v3_large` (`exp07_mobilenet_v3_large_adamw_lr0.0005_bs32_fr0.5`) extracting 128-dimensional pen-kinematic and tremor representations from spiral drawings.
- **Modality 2 — Voice Biomarker Model (Scikit-Learn)**:
  - Trained on the UCI Parkinson's dataset using the top 10 RFE-ExtraTrees selected acoustic features (`MDVP:Fo`, `MDVP:Flo`, `MDVP:Shimmer`, `HNR`, `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`) with a soft-voting ensemble (`0.55 * SVM_RBF + 0.45 * ExtraTrees`).
- **Multimodal Fusion Strategies**:
  - **Early Fusion (Feature-Level)**: Concatenates scaled voice biomarkers with standardized + PCA-compressed spiral image embeddings.
  - **Late Fusion (Decision-Level)**: Combines calibrated probability outputs from the unimodal voice and spiral vision classifiers.
  - **Hybrid Fusion**: Combines scaled voice biomarkers, PCA-compressed visual embeddings, and unimodal probability priors.
- **Single Shared MLflow Run Across DVC Stages**:
  - All 4 DVC pipeline stages attach to the exact same MLflow `run_id` stored in `artifacts/mlflow_run_id.txt` and log to `https://dagshub.com/Parkinson/multiModal.mlflow`.
- **Quality Gate & Schema Serialization**:
  - Models are packaged with an explicit input/output schema dictionary (`.pkl`) and are only promoted/logged if `Test F1-Score > 0.90`.
- **Fault-Tolerant Inference Pipeline**:
  - Supports **Multimodal (`voice + image`)**, **Voice-Only fallback (`image_path=None`)**, and **Image-Only fallback (`voice_data=None`)**.


## Installation & Setup
1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Configure DagsHub Credentials
Export your personal DagsHub username and token so DVC and MLflow can authenticate with the Parkinson/multiModal organization repository:

```bash
export DAGSHUB_USERNAME="<YOUR_DAGSHUB_USERNAME>"
export DAGSHUB_TOKEN="<YOUR_DAGSHUB_TOKEN>"
```
## Running the DVC Pipeline
### Run the Entire End-to-End Pipeline

```bash
rm -f artifacts/mlflow_run_id.txt
dvc repro
```

### Override Stage 1 Download Arguments Manually (Optional)
You can also run Stage 1 directly with custom MLflow artifact paths or local cache overrides:

```bash 

python -m src.stages.01_download_data \
  --config params.yaml \
  --run-id "<SOURCE_IMAGE_MLFLOW_RUN_ID>" \
  --embedding-path "embeddings/spiral_exp07_mobilenet_v3_large_adamw_lr0.0005_bs32_fr0.5_embeddings.pkl" \
  --image-model-path "models_with_schema/spiral_exp07_mobilenet_v3_large_adamw_lr0.0005_bs32_fr0.5_with_schema.pkl"
```


