#from src.inference.download_best_model import download_best_model_for_prediction
from src.inference.predict_pipeline import ParkinsonsMultimodalPredictor as PredictPipeline


#download_best_model_for_prediction()
sample_voice = {
    "MDVP:Fo": 119.992,
    "MDVP:Flo": 74.997,
    "MDVP:Shimmer": 0.04374,
    "HNR": 21.033,
    "RPDE": 0.414783,
    "DFA": 0.815285,
    "spread1": -4.813031,
    "spread2": 0.266482,
    "D2": 2.301442,
    "PPE": 0.284654,
}

pipeline = PredictPipeline()
result = pipeline.predict(voice_data=sample_voice, image_path="sample_spiral_test.png")
print(result)
print("Prediction completed successfully.")