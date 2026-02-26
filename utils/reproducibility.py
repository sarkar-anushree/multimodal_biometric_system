"""
Utility module for ensuring reproducibility across machine learning runs.
"""
import os
import random
import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int) -> None:
    """
    Locks down all random number generators to ensure deterministic execution.

    Args:
        seed (int): The integer seed value to apply across all libraries.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    logger.info(f"Determinism locked: Global random seed set to {seed}")