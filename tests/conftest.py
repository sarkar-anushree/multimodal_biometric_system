import pytest
from omegaconf import OmegaConf

@pytest.fixture
def dummy_cfg():
    """Provides a minimal Hydra DictConfig for component testing."""
    yaml_string = """
    base_path: '/dummy/path'
    num_people: 2
    fingerprint_shape: [128, 128, 3]
    iris_shape: [64, 64, 1]
    num_classes: 2
    dense_units: 16
    dropout_rate: 0.1
    l2_reg: 0.01
    learning_rate: 0.001
    batch_size: 2
    epochs: 1
    mlflow:
      tracking_uri: "http://localhost:5000"
      experiment_name: "test_experiment"
    """
    return OmegaConf.create(yaml_string)