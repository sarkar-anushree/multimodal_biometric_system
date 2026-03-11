"""
Component for reshaping, normalizing, and encoding raw data into model-ready tensors.
"""
import os
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
from omegaconf import DictConfig

from components.base import BaseComponent


class DataPreprocessingComponent(BaseComponent):
    """
    Reads the Parquet dataset, extracts features, applies min-max normalization,
    one-hot encodes labels, and packages the result into an NPZ artifact.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def execute(self, input_parquet_path: str, output_npz_path: str) -> str:
        """
        Executes the data preprocessing pipeline.

        Args:
            input_parquet_path (str): Path to the raw Parquet dataset artifact.
            output_npz_path (str): Destination path for the processed NPZ artifact.

        Returns:
            str: The path to the generated NPZ file.

        Raises:
            FileNotFoundError: If the input Parquet file does not exist.
        """
        self.logger.info("Stage 2: Data Preprocessing Initialization")

        if not os.path.exists(input_parquet_path):
            self.logger.error(f"Missing input artifact: {input_parquet_path}")
            raise FileNotFoundError(f"Missing input artifact: {input_parquet_path}")

        table = pq.read_table(input_parquet_path)

        finger_shape = tuple(self.cfg.fingerprint_shape)
        iris_shape = tuple(self.cfg.iris_shape)

        # Safely extract PyArrow columns to lists, then cast and reshape natively in NumPy
        finger_data = np.array(table.column('fingerprint').to_pylist()).reshape(-1, *finger_shape)
        left_iris_data = np.array(table.column('left_iris').to_pylist()).reshape(-1, *iris_shape)
        right_iris_data = np.array(table.column('right_iris').to_pylist()).reshape(-1, *iris_shape)
        labels = np.array(table.column('label').to_pylist())

        self.logger.info("Applying tensor normalizations and categorical encoding.")
        X_finger = np.repeat(finger_data, 3, axis=-1) / 255.0 if finger_data.shape[-1] == 1 else finger_data / 255.0
        X_left_iris = left_iris_data / 255.0
        X_right_iris = right_iris_data / 255.0

        y = tf.keras.utils.to_categorical(labels)

        os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
        np.savez_compressed(output_npz_path, X_finger=X_finger, X_left=X_left_iris, X_right=X_right_iris, y=y)

        self.logger.info(f"Preprocessing complete. Tensors saved at: {output_npz_path}")
        return output_npz_path
