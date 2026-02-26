"""
Base interface for all machine learning pipeline components.
"""
from abc import ABC, abstractmethod
import logging
from omegaconf import DictConfig
from utils.reproducibility import set_global_seeds


class BaseComponent(ABC):
    """
    Abstract base class enforcing a strict contract for all pipeline components.

    Attributes:
        cfg (DictConfig): The Hydra configuration block specific to the component.
        logger (logging.Logger): A standard logger instance for the component.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the component with its specific configuration.

        Args:
            cfg (DictConfig): Component-specific configuration parameters.
        """
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

        # Lock down determinism the moment the component boots in the container
        if hasattr(self.cfg, 'seed'):
            set_global_seeds(self.cfg.seed)
        else:
            self.logger.warning("No seed found in configuration. Execution may be non-deterministic.")

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        The main execution block of the component.
        Must be overridden by all subclasses.
        """
        pass