import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from components.data_preprocessing_component import DataPreprocessingComponent


def test_data_preprocessing_artifact_creation(dummy_cfg, tmp_path):
    # 1. Setup dummy Parquet file
    finger = np.random.randint(0, 255, (128, 128, 3)).flatten()
    left = np.random.randint(0, 255, (64, 64, 1)).flatten()
    right = np.random.randint(0, 255, (64, 64, 1)).flatten()

    dummy_table = pa.Table.from_arrays(
        [pa.array([finger]), pa.array([left]), pa.array([right]), pa.array([1])],
        names=['fingerprint', 'left_iris', 'right_iris', 'label']
    )
    input_parquet = str(tmp_path / "raw.parquet")
    pq.write_table(dummy_table, input_parquet)

    output_npz = str(tmp_path / "processed.npz")

    # 2. Execute
    preprocessor = DataPreprocessingComponent(dummy_cfg)
    result_path = preprocessor.execute(input_parquet, output_npz)

    # 3. Assertions
    assert result_path == output_npz
    assert os.path.exists(output_npz)

    # Verify contents are normalized
    data = np.load(output_npz)
    assert np.max(data['X_finger']) <= 1.0
    assert data['y'].shape[1] == dummy_cfg.num_classes
