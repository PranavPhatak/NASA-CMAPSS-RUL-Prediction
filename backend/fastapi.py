import os
import pickle
import numpy as np
import tensorflow as tf


class RULPredictor:

    def __init__(self, model_path: str, scaler_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found: {scaler_path}"
            )
        print("Loading LSTM model...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

        print("Loading scaler...")
        with open(scaler_path, "rb") as file:
            self.scaler = pickle.load(file)

        print("Scaler loaded successfully.")

        input_shape = self.model.input_shape

        print(f"Model input shape: {input_shape}")

        self.sequence_length = input_shape[1]
        self.num_features = input_shape[2]

    def validate_input(self, sensor_data):
        data = np.asarray(sensor_data, dtype=np.float32)

        if data.ndim != 2:
            raise ValueError("sensor_data must be a 2D array.")
        
        if data.shape[0] != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} time steps, but received {data.shape[0]}.")

        if data.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, but received {data.shape[1]}.")

        if np.isnan(data).any():
            raise ValueError("Input contains NaN values.")

        if np.isinf(data).any():
            raise ValueError("Input contains infinite values.")

        return data

    def predict(self, sensor_data):

        data = self.validate_input(sensor_data)

        data_scaled = self.scaler.transform(data)

        sequence = np.expand_dims(data_scaled,axis=0)

        prediction = self.model.predict(sequence,verbose=0)

        predicted_rul = float(prediction[0][0])

        predicted_rul = max(predicted_rul, 0.0)

        return round(predicted_rul, 2)