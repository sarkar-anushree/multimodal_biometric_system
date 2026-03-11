import os
import numpy as np
import pyarrow as pa
import ray
from unittest.mock import patch
from components.data_ingestion_component import DataIngestionComponent


@patch('components.data_ingestion_component._process_person_data.remote')
def test_data_ingestion_artifact_creation(mock_process, dummy_cfg, tmp_path):
    ray.init(ignore_reinit_error=True)
    # Create synthetic PyArrow table for mock return
    finger = np.zeros((128, 128, 3)).flatten()
    left = np.zeros((64, 64, 1)).flatten()
    right = np.zeros((64, 64, 1)).flatten()

    dummy_table = pa.Table.from_arrays(
        [pa.array([finger]), pa.array([left]), pa.array([right]), pa.array([0])],
        names=['fingerprint', 'left_iris', 'right_iris', 'label']
    )

    # Wrap the table in a Ray ObjectRef (future) so ray.wait() accepts it
    mock_process.side_effect = lambda *args, **kwargs: ray.put(dummy_table)

    # Define temporary output path
    output_parquet = str(tmp_path / "raw_data.parquet")

    # Execute
    ingestion = DataIngestionComponent(dummy_cfg)
    result_path = ingestion.execute(output_path=output_parquet)

    # Assertions
    assert result_path == output_parquet
    assert os.path.exists(output_parquet)

    # Clean up Ray after the test
    if ray.is_initialized():
        ray.shutdown()
