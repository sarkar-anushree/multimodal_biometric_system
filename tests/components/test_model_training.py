import os
from unittest.mock import patch, MagicMock
from components.model_training_component import ModelTrainingComponent


@patch('components.model_training_component.tf.keras.models.load_model')
@patch('components.model_training_component.np.load')
@patch('components.model_training_component.mlflow')
def test_model_training_artifact_creation(mock_mlflow, mock_np_load, mock_load_model, dummy_cfg, tmp_path):
    # Mock data loading
    mock_np_load.return_value = {
        'X_finger': [1], 'X_left': [1], 'X_right': [1], 'y': [1]
    }

    # Mock model and its history
    mock_model = MagicMock()
    mock_history = MagicMock()
    mock_history.history = {'loss': [0.9], 'accuracy': [0.5]}
    mock_model.fit.return_value = mock_history
    mock_load_model.return_value = mock_model

    # Define paths
    input_npz = str(tmp_path / "in.npz")
    input_model = str(tmp_path / "in.h5")
    output_model = str(tmp_path / "out.h5")
    output_history = str(tmp_path / "history.json")

    # Create dummy input files so the os.path.exists checks pass
    open(input_npz, 'a').close()
    open(input_model, 'a').close()

    # Execute
    trainer = ModelTrainingComponent(dummy_cfg)
    res_model, res_hist = trainer.execute(input_npz, input_model, output_model, output_history)

    # Assertions
    assert res_model == output_model
    assert res_hist == output_history
    assert os.path.exists(output_history)
    mock_model.save.assert_called_once_with(output_model)