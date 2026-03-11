import os
import json
import numpy as np
from unittest.mock import patch, MagicMock
from components.model_training_component import ModelTrainingComponent


# Patch MLflow so it doesn't try to connect to the internet during the test
@patch('mlflow.log_artifact')
@patch('mlflow.tensorflow.autolog')
@patch('mlflow.start_run')
@patch('mlflow.set_experiment')
@patch('mlflow.set_tracking_uri')
@patch('tensorflow.keras.backend.set_value')
@patch('tensorflow.keras.models.load_model')
def test_model_training_artifact_creation(
        mock_load_model, mock_set_value, mock_set_uri, mock_set_exp,
        mock_start_run, mock_autolog, mock_log_artifact, dummy_cfg, tmp_path
):
    # --- SETUP CONFIG ---
    dummy_cfg.seed = 42
    dummy_cfg.learning_rate = 0.001
    dummy_cfg.batch_size = 8
    dummy_cfg.epochs = 2
    dummy_cfg.mlflow.tracking_uri = "http://localhost:5000"
    dummy_cfg.mlflow.experiment_name = "test_exp"

    # --- SETUP DUMMY INPUT FILES ---
    input_npz = str(tmp_path / "processed_data.npz")
    np.savez_compressed(
        input_npz,
        X_finger=np.zeros((1, 128, 128, 3)),
        X_left=np.zeros((1, 64, 64, 1)),
        X_right=np.zeros((1, 64, 64, 1)),
        y=np.zeros((1, 5))
    )

    input_model = str(tmp_path / "untrained_model.h5")
    with open(input_model, 'w') as f:
        f.write("fake_model_weights")

    output_model = str(tmp_path / "trained_model.h5")
    output_history = str(tmp_path / "history.json")

    # --- SETUP MOCK MODEL ---
    mock_model = MagicMock()
    mock_history = MagicMock()
    # Provide fake metrics so the JSON dump succeeds
    mock_history.history = {'accuracy': [0.85, 0.90], 'loss': [0.5, 0.3]}
    mock_model.fit.return_value = mock_history

    # Force load_model to return OUR mock_model
    mock_load_model.return_value = mock_model

    # --- EXECUTE ---
    trainer = ModelTrainingComponent(dummy_cfg)
    res_model, res_hist = trainer.execute(input_npz, input_model, output_model, output_history)

    # --- ASSERTIONS ---
    # Now we check that OUR specific mock_model was saved!
    mock_model.fit.assert_called_once()
    mock_model.save.assert_called_once_with(output_model)

    assert res_model == output_model
    assert res_hist == output_history
    assert os.path.exists(output_history)  # Check that the history JSON was actually written