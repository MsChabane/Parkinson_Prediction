import pickle
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
import mlflow

from src.config.mlflow_helper import load_params, get_or_create_shared_run
from src.utils.vision_arch import safe_load_pickle
from src.utils.logger import get_logger


def build_feature_matrices(split_dict: dict) -> dict:
    early = np.hstack([split_dict["img_pca"], split_dict["voc_feat"]]).astype(np.float32)
    p_i = split_dict["img_prob"].reshape(-1, 1)
    p_v = split_dict["voc_prob"].reshape(-1, 1)
    late = np.hstack([p_i, p_v, p_i * p_v, np.abs(p_i - p_v)]).astype(np.float32)
    hybrid = np.hstack([early, late]).astype(np.float32)
    return {"early_fusion": early, "late_fusion": late, "hybrid_fusion": hybrid}


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thresh, best_f1 = 0.50, -1.0
    for thresh in np.linspace(0.25, 0.75, 51):
        score = f1_score(y_true, (y_prob >= thresh).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return round(best_thresh, 4)


def main():
    logger = get_logger("train_multimodal")
    params = load_params()
    seed = params["seed"]
    tr_cfg = params["train"]

    raw_dir = Path(params["download"]["raw_dir"])
    proc_dir = Path("data/processed")
    prep_dir = Path("artifacts/preprocessors")
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    paired_splits = safe_load_pickle(proc_dir / "paired_multimodal_splits.pkl")
    image_bundle = safe_load_pickle(raw_dir / "spiral_image_model_with_schema.pkl")
    voice_bundle = safe_load_pickle(prep_dir / "unimodal_voice_models.pkl")
    voice_scaler = safe_load_pickle(prep_dir / "voice_scaler.pkl")
    img_emb_scaler = safe_load_pickle(prep_dir / "img_emb_scaler.pkl")
    img_pca = safe_load_pickle(prep_dir / "img_pca.pkl")

    mm_train, mm_val, mm_test = paired_splits["train"], paired_splits["validation"], paired_splits["testing"]
    X_train_map = build_feature_matrices(mm_train)
    X_val_map = build_feature_matrices(mm_val)
    X_test_map = build_feature_matrices(mm_test)

    candidate_specs = [
        ("early_fusion", "svm_rbf", SVC(kernel="rbf", C=tr_cfg["svm_c"], gamma="scale", probability=True, random_state=seed)),
        ("early_fusion", "extra_trees", ExtraTreesClassifier(n_estimators=tr_cfg["et_estimators"], random_state=seed)),
        ("early_fusion", "random_forest", RandomForestClassifier(n_estimators=tr_cfg["rf_estimators"], random_state=seed)),
        ("early_fusion", "mlp_net", MLPClassifier(hidden_layer_sizes=tuple(tr_cfg["mlp_hidden"]), max_iter=500, random_state=seed)),
        ("late_fusion", "stacking_logreg", LogisticRegression(C=1.0, random_state=seed)),
        ("late_fusion", "stacking_et", ExtraTreesClassifier(n_estimators=200, max_depth=4, random_state=seed)),
        ("hybrid_fusion", "svm_rbf", SVC(kernel="rbf", C=tr_cfg["svm_c"], gamma="scale", probability=True, random_state=seed)),
        ("hybrid_fusion", "extra_trees", ExtraTreesClassifier(n_estimators=tr_cfg["et_estimators"], random_state=seed)),
        ("hybrid_fusion", "gradient_boosting", GradientBoostingClassifier(n_estimators=tr_cfg["gb_estimators"], learning_rate=tr_cfg["gb_lr"], random_state=seed)),
    ]

    trained_candidates = {}
    best_tag = None
    best_val_score = -1.0

    for idx, (f_type, clf_name, clf) in enumerate(candidate_specs, start=1):
        tag = f"mm{idx:02d}_{f_type}_{clf_name}"
        clf.fit(X_train_map[f_type], mm_train["labels"])

        val_probs = clf.predict_proba(X_val_map[f_type])[:, 1]
        test_probs = clf.predict_proba(X_test_map[f_type])[:, 1]

        opt_thresh = find_optimal_threshold(mm_val["labels"], val_probs)
        val_f1 = f1_score(mm_val["labels"], (val_probs >= opt_thresh).astype(int), zero_division=0)
        val_auc = roc_auc_score(mm_val["labels"], val_probs)
        composite = val_f1 + val_auc

        trained_candidates[tag] = {
            "tag": tag,
            "fusion_type": f_type,
            "classifier_name": clf_name,
            "model": clf,
            "threshold": opt_thresh,
            "val_probs": val_probs,
            "test_probs": test_probs,
            "val_f1": float(val_f1),
            "val_auc": float(val_auc),
        }

        if composite > best_val_score:
            best_val_score = composite
            best_tag = tag

    all_candidates_path = models_dir / "all_multimodal_candidates.pkl"
    with open(all_candidates_path, "wb") as f:
        pickle.dump(trained_candidates, f)

    winner = trained_candidates[best_tag]
    image_schema = image_bundle["schema"]
    unimodal_img_thresh = image_schema["output_schema"]["threshold"]
    unimodal_voc_thresh = find_optimal_threshold(mm_val["labels"], mm_val["voc_prob"])

    multimodal_schema = {
        "model_name": f"parkinsons_{best_tag}",
        "fusion_type": winner["fusion_type"],
        "classifier": winner["classifier_name"],
        "voice_required_features": voice_bundle["features"],
        "voice_column_aliases": voice_bundle["aliases"],
        "image_input_schema": image_schema["input_schema"],
        "thresholds": {
            "multimodal": winner["threshold"],
            "image_only_fallback": unimodal_img_thresh,
            "voice_only_fallback": unimodal_voc_thresh,
        },
        "class_mapping": {"healthy": 0, "parkinson": 1},
        "inverse_class_mapping": {0: "healthy", 1: "parkinson"},
    }

    production_bundle = {
        "schema": multimodal_schema,
        "fusion_model": winner["model"],
        "img_emb_scaler": img_emb_scaler,
        "img_pca": img_pca,
        "image_state_dict": image_bundle["state_dict"],
        "image_hyperparameters": image_schema["hyperparameters"],
        "voice_scaler": voice_scaler,
        "voice_svm": voice_bundle["voice_svm"],
        "voice_et": voice_bundle["voice_et"],
        "voice_weights": voice_bundle["weights"],
    }

    best_bundle_path = models_dir / "multimodal_best_with_schema.pkl"
    with open(best_bundle_path, "wb") as f:
        pickle.dump(production_bundle, f)

    with get_or_create_shared_run(params, create_new=False):
        mlflow.set_tags({
            "mm_best_tag": best_tag,
            "mm_best_fusion_type": winner["fusion_type"],
            "mm_best_classifier": winner["classifier_name"],
        })
        logger.info(f"Stage 3 Complete | Best Multimodal Model: {best_tag} (Val F1={winner['val_f1']:.4f}, Val AUC={winner['val_auc']:.4f})")


if __name__ == "__main__":
    main()