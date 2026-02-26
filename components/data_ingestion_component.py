"""
Component for parallelized reading of raw biometric image files into Parquet artifacts.
"""
import os
import logging
import ray
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from omegaconf import DictConfig

from components.base import BaseComponent

logger = logging.getLogger(__name__)


def _load_image(person_path: str, modality: str, size: tuple, grayscale: bool = False) -> np.ndarray:
    """
    Loads and converts a biometric image from disk to a NumPy array.

    Args:
        person_path (str): The root directory for the specific person.
        modality (str): The subfolder name (e.g., 'Fingerprint', 'left', 'right').
        size (tuple): The target (width, height) to resize the image to.
        grayscale (bool, optional): Whether to load the image in grayscale. Defaults to False.

    Returns:
        np.ndarray: The processed image array, or None if the path/file does not exist.
    """
    modality_path = os.path.join(person_path, modality)
    if not os.path.exists(modality_path):
        return None
    files = sorted([f for f in os.listdir(modality_path) if f.endswith(('.bmp', '.BMP'))])
    if not files:
        return None
    color_mode = 'grayscale' if grayscale else 'rgb'
    img = load_img(os.path.join(modality_path, files[0]), color_mode=color_mode, target_size=size)
    return img_to_array(img)


@ray.remote
def _process_person_data(person_id: int, base_path: str, finger_size: tuple, iris_size: tuple) -> pa.Table:
    """
    Ray distributed task to process all modalities for a single person.

    Args:
        person_id (int): The numerical ID of the person.
        base_path (str): The root dataset directory.
        finger_size (tuple): Target size for fingerprint images.
        iris_size (tuple): Target size for iris images.

    Returns:
        pa.Table: A PyArrow table containing the flattened image arrays and label,
                  or None if data is incomplete.
    """
    person_path = os.path.join(base_path, str(person_id))
    if not os.path.exists(person_path):
        return None

    finger_img = _load_image(person_path, 'Fingerprint', finger_size, grayscale=False)
    left_img = _load_image(person_path, 'left', iris_size, grayscale=True)
    right_img = _load_image(person_path, 'right', iris_size, grayscale=True)

    if finger_img is None or left_img is None or right_img is None:
        return None

    label = person_id - 1
    table = pa.Table.from_arrays(
        [pa.array([finger_img.flatten()]), pa.array([left_img.flatten()]),
         pa.array([right_img.flatten()]), pa.array([label])],
        names=['fingerprint', 'left_iris', 'right_iris', 'label']
    )
    return table


class DataIngestionComponent(BaseComponent):
    """
    Orchestrates the data ingestion process, leveraging Ray to process files
    concurrently and outputting a compressed Parquet dataset.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def execute(self, output_path: str) -> str:
        """
        Executes the ingestion pipeline.

        Args:
            output_path (str): The destination file path for the Parquet artifact.

        Returns:
            str: The path to the generated Parquet file.

        Raises:
            ValueError: If no valid data could be loaded from the source directory.
        """
        self.logger.info(f"Stage 1: Data Ingestion (Target artifact: {output_path})")
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        finger_size = tuple(self.cfg.fingerprint_shape[:2])
        iris_size = tuple(self.cfg.iris_shape[:2])

        futures = [_process_person_data.remote(pid, self.cfg.base_path, finger_size, iris_size)
                   for pid in range(1, self.cfg.num_people + 1)]

        results, unready = [], futures
        with tqdm(total=len(futures), desc="Reading files") as pbar:
            while unready:
                ready, unready = ray.wait(unready, num_returns=1)
                results.extend(ray.get(ready))
                pbar.update(len(ready))

        valid_tables = [res for res in results if res is not None]
        if ray.is_initialized():
            ray.shutdown()

        if not valid_tables:
            self.logger.error("No valid data loaded. Check the base dataset path.")
            raise ValueError("No data loaded.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pq.write_table(pa.concat_tables(valid_tables), output_path)
        self.logger.info(f"Ingestion complete. Artifact saved successfully.")

        return output_path
