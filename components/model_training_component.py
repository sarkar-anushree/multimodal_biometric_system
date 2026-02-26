"""
Component for executing model training while automatically logging telemetry to MLflow.
"""
import os
import json
import logging
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from omegaconf import DictConfig

from components.base import BaseComponent


class ModelTrainingComponent(BaseComponent):
    """
    Loads preprocessed data and an untrained architecture, executes the fit loop,
    and bridges telemetry data directly to the MLflow tracking server.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def execute(self, input_npz_path: str, input_model_path: str, output_model_path: str,
                output_history_path: str) -> tuple:
        """
        Executes the training sequence.

        Args:
            input_npz_path (str): Path to the processed tensors.
            input_model_path (str): Path to the untrained `.h5` model artifact.
            output_model_path (str): Destination for the finalized `.h5` model weights.
            output_history_path (str): Destination for the training metrics JSON.

        Returns:
            tuple: Paths to the finalized model and history artifacts.

        Raises:
            FileNotFoundError: If input datasets or models are missing.
        """
        self.logger.info("Stage 4: Initializing MLflow Model Training")

        if not os.path.exists(input_npz_path) or not os.path.exists(input_model_path):
            self.logger.error("Missing critical input artifacts for training stage.")
            raise FileNotFoundError("Missing input artifacts for training stage.")

        data = np.load(input_npz_path)
        X_finger, X_left, X_right, y = data['X_finger'], data['X_left'], data['X_right'], data['y']
        model = tf.keras.models.load_model(input_model_path)

        tf.keras.backend.set_value(model.optimizer.learning_rate, self.cfg.learning_rate)

        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        self.logger.info(f"Connected to MLflow Tracking URI: {self.cfg.mlflow.tracking_uri}")

        with mlflow.start_run(run_name="kfp_training_job"):
            mlflow.log_params({
                "learning_rate": self.cfg.learning_rate,
                "batch_size": self.cfg.batch_size,
                "epochs": self.cfg.epochs
            })
            mlflow.tensorflow.autolog()

            self.logger.info(f"Initiating fit routine over {self.cfg.epochs} epochs.")
            history = model.fit(
                [X_finger, X_left, X_right], y,
                batch_size=self.cfg.batch_size,
                epochs=self.cfg.epochs,
                verbose=1
            )

            serializable_history = {metric: [float(val) for val in values] for metric, values in
                                    history.history.items()}
            os.makedirs(os.path.dirname(output_history_path), exist_ok=True)
            with open(output_history_path, 'w') as f:
                json.dump(serializable_history, f, indent=4)
            mlflow.log_artifact(output_history_path, artifact_path="custom_metrics")

        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        model.save(output_model_path)
        self.logger.info(f"Training finalized. Model securely exported to: {output_model_path}")

        return output_model_path, output_history_path