import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import mlflow

from src.config.mlflow_helper import load_params, get_or_create_shared_run
from src.utils.vision_arch import safe_load_pickle
from src.utils.logger import get_logger

logger = get_logger("build_pairs_and_preprocess")
COLUMN_ALIASES = {
    "MDVP:Fo(Hz)": "MDVP:Fo",
    "MDVP:Fhi(Hz)": "MDVP:Fhi",
    "MDVP:Flo(Hz)": "MDVP:Flo",
}


def pair_modalities_by_class(
    img_split: dict,
    v_features: np.ndarray,
    v_probs: np.ndarray,
    v_labels: np.ndarray,
    seed: int,
    noise_std: float,
    is_train: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    img_emb = img_split["embeddings"]
    img_prob = img_split["probs"]
    img_lbl = img_split["labels"].astype(np.int64)

    p_img_emb, p_img_prob = [], []
    p_voc_feat, p_voc_prob = [], []
    p_labels = []

    for cls in [0, 1]:
        img_idx = np.where(img_lbl == cls)[0]
        voc_idx = np.where(v_labels == cls)[0]

        for i, i_idx in enumerate(img_idx):
            v_i = voc_idx[i % len(voc_idx)]
            v_f = v_features[v_i].copy()
            if is_train and i >= len(voc_idx) and noise_std > 0:
                v_f = v_f + rng.normal(0.0, noise_std, size=v_f.shape).astype(np.float32)

            p_img_emb.append(img_emb[i_idx])
            p_img_prob.append(img_prob[i_idx])
            p_voc_feat.append(v_f)
            p_voc_prob.append(v_probs[v_i])
            p_labels.append(cls)

    return {
        "img_emb": np.array(p_img_emb, dtype=np.float32),
        "img_prob": np.array(p_img_prob, dtype=np.float32),
        "voc_feat": np.array(p_voc_feat, dtype=np.float32),
        "voc_prob": np.array(p_voc_prob, dtype=np.float32),
        "labels": np.array(p_labels, dtype=np.int64),
    }


def main():
    
    params = load_params()
    seed = params["seed"]
    prep_cfg = params["preprocess"]
    raw_dir = Path(params["download"]["raw_dir"])

    proc_dir = Path("data/processed")
    prep_art_dir = Path("artifacts/preprocessors")
    proc_dir.mkdir(parents=True, exist_ok=True)
    prep_art_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings = safe_load_pickle(raw_dir / "spiral_embeddings.pkl")
    voice_df = pd.read_csv(raw_dir / "voice_parkinsons.csv").rename(columns=COLUMN_ALIASES)

    selected_features = prep_cfg["selected_voice_features"]
    X_voice = voice_df[selected_features].values.astype(np.float32)
    y_voice = voice_df["status"].values.astype(np.int64)

    X_v_tr_val, X_v_test, y_v_tr_val, y_v_test = train_test_split(
        X_voice, y_voice, test_size=prep_cfg["test_size"], stratify=y_voice, random_state=seed
    )
    X_v_train, X_v_val, y_v_train, y_v_val = train_test_split(
        X_v_tr_val, y_v_tr_val, test_size=prep_cfg["val_size"], stratify=y_v_tr_val, random_state=seed
    )

    voice_scaler = StandardScaler()
    X_v_train_sc = voice_scaler.fit_transform(X_v_train)
    X_v_val_sc = voice_scaler.transform(X_v_val)
    X_v_test_sc = voice_scaler.transform(X_v_test)

    voice_svm = SVC(kernel="rbf", C=prep_cfg["voice_svm_c"], gamma="scale", probability=True, random_state=seed)
    voice_et = ExtraTreesClassifier(n_estimators=prep_cfg["voice_et_estimators"], random_state=seed)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_tr_svm = cross_val_predict(voice_svm, X_v_train_sc, y_v_train, cv=cv, method="predict_proba")[:, 1]
    p_tr_et = cross_val_predict(voice_et, X_v_train_sc, y_v_train, cv=cv, method="predict_proba")[:, 1]
    w_svm, w_et = prep_cfg["voice_svm_weight"], prep_cfg["voice_et_weight"]
    p_v_train = w_svm * p_tr_svm + w_et * p_tr_et

    voice_svm.fit(X_v_train_sc, y_v_train)
    voice_et.fit(X_v_train_sc, y_v_train)

    p_v_val = w_svm * voice_svm.predict_proba(X_v_val_sc)[:, 1] + w_et * voice_et.predict_proba(X_v_val_sc)[:, 1]
    p_v_test = w_svm * voice_svm.predict_proba(X_v_test_sc)[:, 1] + w_et * voice_et.predict_proba(X_v_test_sc)[:, 1]

    mm_train = pair_modalities_by_class(image_embeddings["train"], X_v_train_sc, p_v_train, y_v_train, seed, prep_cfg["pair_noise_std"], is_train=True)
    mm_val = pair_modalities_by_class(image_embeddings["validation"], X_v_val_sc, p_v_val, y_v_val, seed, 0.0, is_train=False)
    mm_test = pair_modalities_by_class(image_embeddings["testing"], X_v_test_sc, p_v_test, y_v_test, seed, 0.0, is_train=False)

    img_emb_scaler = StandardScaler()
    img_pca = PCA(n_components=prep_cfg["pca_components"], random_state=seed)

    mm_train["img_pca"] = img_pca.fit_transform(img_emb_scaler.fit_transform(mm_train["img_emb"])).astype(np.float32)
    mm_val["img_pca"] = img_pca.transform(img_emb_scaler.transform(mm_val["img_emb"])).astype(np.float32)
    mm_test["img_pca"] = img_pca.transform(img_emb_scaler.transform(mm_test["img_emb"])).astype(np.float32)

    with open(prep_art_dir / "voice_scaler.pkl", "wb") as f:
        pickle.dump(voice_scaler, f)
    with open(prep_art_dir / "img_emb_scaler.pkl", "wb") as f:
        pickle.dump(img_emb_scaler, f)
    with open(prep_art_dir / "img_pca.pkl", "wb") as f:
        pickle.dump(img_pca, f)
    with open(prep_art_dir / "unimodal_voice_models.pkl", "wb") as f:
        pickle.dump({"voice_svm": voice_svm, "voice_et": voice_et, "weights": (w_svm, w_et), "features": selected_features, "aliases": COLUMN_ALIASES}, f)

    paired_data_path = proc_dir / "paired_multimodal_splits.pkl"
    with open(paired_data_path, "wb") as f:
        pickle.dump({"train": mm_train, "validation": mm_val, "testing": mm_test}, f)

    with get_or_create_shared_run(params, create_new=False):
        mlflow.log_params({
            "prep_pca_components": prep_cfg["pca_components"],
            "prep_pca_explained_variance": round(float(np.sum(img_pca.explained_variance_ratio_)), 4),
            "prep_num_voice_features": len(selected_features),
            "prep_paired_train_samples": len(mm_train["labels"]),
            "prep_paired_val_samples": len(mm_val["labels"]),
            "prep_paired_test_samples": len(mm_test["labels"]),
        })
        mlflow.log_artifacts(str(prep_art_dir), artifact_path="preprocessors")
        logger.info("Stage 2 Complete | Stored Scalers, PCA, and Paired Multimodal Dataset.")


if __name__ == "__main__":
    main()