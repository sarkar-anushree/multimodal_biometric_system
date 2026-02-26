"""
Component for defining and compiling the multimodal Keras architecture.
"""
import os
import logging
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from omegaconf import DictConfig

from components.base import BaseComponent


class ModelBuildingComponent(BaseComponent):
    """
    Constructs an untrained multi-branch neural network utilizing a frozen
    MobileNetV2 backbone for fingerprints and custom CNN blocks for irises.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _create_iris_branch(self, input_shape: tuple, name_prefix: str = "") -> Model:
        """
        Creates a dedicated Convolutional Neural Network branch for iris processing.

        Args:
            input_shape (tuple): The (height, width, channels) of the iris image.
            name_prefix (str, optional): Prefix for layer naming. Defaults to "".

        Returns:
            Model: A compiled Keras Model representing the iris branch.
        """
        inputs = layers.Input(shape=input_shape, name=f"{name_prefix}_input")
        x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        return Model(inputs, x, name=f"{name_prefix}_branch")

    def execute(self, output_model_path: str) -> str:
        """
        Builds the architecture and saves it to disk as an `.h5` artifact.

        Args:
            output_model_path (str): The destination path for the untrained model.

        Returns:
            str: The path to the saved untrained model artifact.
        """
        self.logger.info("Stage 3: Defining Multimodal Network Architecture")

        finger_shape = tuple(self.cfg.fingerprint_shape)
        iris_shape = tuple(self.cfg.iris_shape)

        base_model = MobileNetV2(input_shape=finger_shape, include_top=False, weights='imagenet', pooling='avg')
        base_model.trainable = False

        iris_processor = self._create_iris_branch(iris_shape)

        fingerprint_input = layers.Input(shape=finger_shape, name="fingerprint_input")
        left_iris_input = layers.Input(shape=iris_shape, name="left_iris_input")
        right_iris_input = layers.Input(shape=iris_shape, name="right_iris_input")

        combined = layers.Concatenate(name="feature_concat")([
            base_model(fingerprint_input),
            iris_processor(left_iris_input),
            iris_processor(right_iris_input)
        ])

        x = layers.Dense(self.cfg.dense_units, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(self.cfg.l2_reg))(combined)
        x = layers.Dropout(self.cfg.dropout_rate)(x)
        output = layers.Dense(self.cfg.num_classes, activation='softmax')(x)

        model = Model(inputs=[fingerprint_input, left_iris_input, right_iris_input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        model.save(output_model_path)
        self.logger.info(f"Untrained architecture saved at: {output_model_path}")

        return output_model_path
