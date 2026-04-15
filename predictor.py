import json

import joblib
import pandas as pd
from tensorflow.keras.models import load_model


class RankingPredictor:
    def __init__(self, ml_model_path, dl_model_path, preprocessor_path, metadata_path):
        # Load ML model
        self.ml_model = joblib.load(ml_model_path)

        # Load preprocessor
        self.preprocessor = joblib.load(preprocessor_path)

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load DL model safely
        self.dl_model = None
        try:
            self.dl_model = load_model(
                dl_model_path,
                compile=False  # prevents mse error
            )

            # Manually compile for inference compatibility.
            self.dl_model.compile(
                optimizer="adam",
                loss="mean_squared_error",
                metrics=["mean_absolute_error"],
            )

            print("Deep Learning model loaded successfully.")

        except Exception as e:
            print("DL model loading failed. Using ML model instead.")
            print("Reason:", e)

    def prepare_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure input has the required columns and order."""
        for col in self.metadata["feature_columns"]:
            if col not in input_df.columns:
                input_df[col] = 0

        return input_df[self.metadata["feature_columns"]]

    def predict(self, input_df: pd.DataFrame):
        """Predict ranking scores using DL when available, else ML."""
        input_df = self.prepare_input(input_df)

        if self.dl_model is not None:
            try:
                processed = self.preprocessor.transform(input_df)

                if hasattr(processed, "toarray"):
                    processed = processed.toarray()

                predictions = self.dl_model.predict(processed).flatten()
                return predictions, "DL"

            except Exception as e:
                print("DL prediction failed. Switching to ML model.")
                print("Reason:", e)

        predictions = self.ml_model.predict(input_df)
        return predictions, "ML"
