
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

from src.utils.vision_arch import ConfigurableVisionClassifier, safe_load_pickle


class ParkinsonsMultimodalPredictor:
    def __init__(self, bundle_path: str = "artifacts/models/multimodal_best_with_schema.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = safe_load_pickle(bundle_path)
        self.schema = self.bundle["schema"]
        self.vision_model = None

    def _get_vision_model(self):
        if self.vision_model is None:
            hp = self.bundle["image_hyperparameters"]
            net = ConfigurableVisionClassifier(
                base_model_name=hp["base_model"],
                pretrained=False,
                embedding_dim=hp["embedding_dim"],
                dropout_rate=hp["dropout_rate"],
                freeze_ratio=hp["freeze_ratio"],
            ).to(self.device)
            net.load_state_dict(self.bundle["image_state_dict"])
            net.eval()
            self.vision_model = net
        return self.vision_model

    def predict(self, voice_data: dict = None, image_path: str = None) -> dict:
        if voice_data is None and image_path is None:
            raise ValueError("Provide at least one modality: voice_data or image_path.")

        inv_map = self.schema["inverse_class_mapping"]
        thresholds = self.schema["thresholds"]

        p_voice, v_scaled = None, None
        p_image, img_pca_vec = None, None

        if voice_data is not None:
            df_v = pd.DataFrame([voice_data]).rename(columns=self.schema["voice_column_aliases"])
            v_raw = df_v[self.schema["voice_required_features"]].values.astype(np.float32)
            v_scaled = self.bundle["voice_scaler"].transform(v_raw)
            p_svm = float(self.bundle["voice_svm"].predict_proba(v_scaled)[0, 1])
            p_et = float(self.bundle["voice_et"].predict_proba(v_scaled)[0, 1])
            w_svm, w_et = self.bundle["voice_weights"]
            p_voice = w_svm * p_svm + w_et * p_et

        if image_path is not None:
            net = self._get_vision_model()
            in_schema = self.schema["image_input_schema"]
            img_size = in_schema["tensor_shape"][2]
            tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=in_schema["normalization_mean"], std=in_schema["normalization_std"]),
            ])
            img_tensor = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = net.extract_embedding(img_tensor)
                logit = net.classifier(emb).squeeze(-1)
                p_image = float(torch.sigmoid(logit).item())
                emb_np = emb.cpu().numpy()

            img_pca_vec = self.bundle["img_pca"].transform(self.bundle["img_emb_scaler"].transform(emb_np))

        if voice_data is not None and image_path is not None:
            f_type = self.schema["fusion_type"]
            mode = f"multimodal_{f_type}"
            early_vec = np.hstack([img_pca_vec, v_scaled]).astype(np.float32)
            late_vec = np.array([[p_image, p_voice, p_image * p_voice, abs(p_image - p_voice)]], dtype=np.float32)

            if f_type == "early_fusion":
                x_in = early_vec
            elif f_type == "late_fusion":
                x_in = late_vec
            else:
                x_in = np.hstack([early_vec, late_vec]).astype(np.float32)

            final_prob = float(self.bundle["fusion_model"].predict_proba(x_in)[0, 1])
            thresh = thresholds["multimodal"]
        elif voice_data is not None:
            mode = "unimodal_voice_fallback"
            final_prob = p_voice
            thresh = thresholds["voice_only_fallback"]
        else:
            mode = "unimodal_image_fallback"
            final_prob = p_image
            thresh = thresholds["image_only_fallback"]

        pred = int(final_prob >= thresh)
        return {
            "mode_used": mode,
            "prediction": pred,
            "diagnosis": inv_map[pred],
            "final_probability_pd": round(final_prob, 4),
            "threshold_used": thresh,
            "voice_probability_pd": round(p_voice, 4) if p_voice is not None else None,
            "image_probability_pd": round(p_image, 4) if p_image is not None else None,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-path", type=str, default="artifacts/models/multimodal_best_with_schema.pkl")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--voice-json", type=str, default=None)
    args = parser.parse_args()

    v_dict = json.loads(args.voice_json) if args.voice_json else None
    predictor = ParkinsonsMultimodalPredictor(bundle_path=args.bundle_path)
    result = predictor.predict(voice_data=v_dict, image_path=args.image_path)
    print(json.dumps(result, indent=4))