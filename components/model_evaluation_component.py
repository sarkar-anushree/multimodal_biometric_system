"""
Component for parsing historical metrics and generating evaluation visuals.
"""
import os
import json
import logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from components.base import BaseComponent


class ModelEvaluationComponent(BaseComponent):
    """
    Parses standard JSON training history to output graphical performance artifacts.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def execute(self, input_history_path: str, output_plot_path: str) -> str:
        """
        Renders the loss and accuracy curves based on Keras history.

        Args:
            input_history_path (str): Path to the training history JSON.
            output_plot_path (str): Destination path for the generated PNG.

        Returns:
            str: Path to the successfully generated plot artifact.

        Raises:
            FileNotFoundError: If the history JSON is unavailable.
        """
        self.logger.info("Stage 5: Starting Metric Evaluation & Visualization")

        if not os.path.exists(input_history_path):
            self.logger.error(f"Failed to load history artifact from: {input_history_path}")
            raise FileNotFoundError(f"Missing history artifact: {input_history_path}")

        with open(input_history_path, 'r') as f:
            history_dict = json.load(f)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history_dict['accuracy'], label='Accuracy', color='blue')
        plt.title('Training Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history_dict['loss'], label='Loss', color='red')
        plt.title('Training Loss')
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        plt.savefig(output_plot_path)

        self.logger.info(f"Evaluation plot rendered and saved to: {output_plot_path}")
        return output_plot_path