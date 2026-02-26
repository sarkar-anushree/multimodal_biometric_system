import os
import json
from components.model_evaluation_component import ModelEvaluationComponent


def test_model_evaluation_artifact_creation(dummy_cfg, tmp_path):
    # 1. Setup dummy JSON history
    input_history = str(tmp_path / "history.json")
    with open(input_history, 'w') as f:
        json.dump({'loss': [0.9, 0.4], 'accuracy': [0.5, 0.8]}, f)

    output_plot = str(tmp_path / "plot.png")

    # 2. Execute
    evaluator = ModelEvaluationComponent(dummy_cfg)
    result_path = evaluator.execute(input_history, output_plot)

    # 3. Assertions
    assert result_path == output_plot
    assert os.path.exists(output_plot)
