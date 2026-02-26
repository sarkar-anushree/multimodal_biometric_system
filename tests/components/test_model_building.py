import os
import tensorflow as tf
from components.model_building_component import ModelBuildingComponent


def test_model_building_artifact_creation(dummy_cfg, tmp_path):
    output_model = str(tmp_path / "untrained_model.h5")

    # Execute
    builder = ModelBuildingComponent(dummy_cfg)
    result_path = builder.execute(output_model)

    # Assertions
    assert result_path == output_model
    assert os.path.exists(output_model)

    # Verify it can be loaded and has correct structure
    loaded_model = tf.keras.models.load_model(output_model)
    assert len(loaded_model.inputs) == 3
    assert loaded_model.output_shape == (None, dummy_cfg.num_classes)
