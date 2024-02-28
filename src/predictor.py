import joblib

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils


class CprPredictor(Predictor):

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        prediction_utils.download_model_artifacts(artifacts_uri)
        self._model = joblib.load("model.joblib")

    def predict(self, instances):
        print("CUSTOM PREDICTION")
        predictions = self._model.predict(instances["instances"])
        probabilities = self._model.predict_proba(instances["instances"])

        unknown_mask = (probabilities.max(axis=1) < 0.3)
        predictions[unknown_mask] = 'unknown'

        pred_list = predictions.tolist()
        proba_list = probabilities.max(axis=1).tolist()

        classifications = []
        for prediction, probability in zip(pred_list, proba_list):
            classifications.append({"category": prediction, "probability": probability})
        return {"predictions": classifications}