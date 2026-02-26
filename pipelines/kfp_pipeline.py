"""
The primary orchestrator mapping the object-oriented Python components to Kubeflow containers.
"""
import json
import logging
from kfp import dsl
from omegaconf import OmegaConf

from pipeline.base import BasePipeline

BASE_IMAGE = "biometric-pipeline:v1"

# Setup root logger for execution environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# Define KFP Components
# ==========================================

@dsl.component(base_image=BASE_IMAGE)
def ingest_op(cfg_json: str, output_parquet: dsl.OutputPath("Dataset")):
    import json
    from omegaconf import OmegaConf
    from components.data_ingestion_component import DataIngestionComponent

    cfg = OmegaConf.create(json.loads(cfg_json))
    DataIngestionComponent(cfg).execute(output_path=output_parquet)


@dsl.component(base_image=BASE_IMAGE)
def preprocess_op(cfg_json: str, input_parquet: dsl.InputPath("Dataset"), output_npz: dsl.OutputPath("Dataset")):
    import json
    from omegaconf import OmegaConf
    from components.data_preprocessing_component import DataPreprocessingComponent

    cfg = OmegaConf.create(json.loads(cfg_json))
    DataPreprocessingComponent(cfg).execute(input_parquet_path=input_parquet, output_npz_path=output_npz)


@dsl.component(base_image=BASE_IMAGE)
def build_model_op(cfg_json: str, output_model: dsl.OutputPath("Model")):
    import json
    from omegaconf import OmegaConf
    from components.model_building_component import ModelBuildingComponent

    cfg = OmegaConf.create(json.loads(cfg_json))
    ModelBuildingComponent(cfg).execute(output_model_path=output_model)


@dsl.component(base_image=BASE_IMAGE)
def train_model_op(
        cfg_json: str,
        input_npz: dsl.InputPath("Dataset"),
        input_model: dsl.InputPath("Model"),
        output_model: dsl.OutputPath("Model"),
        output_history: dsl.OutputPath("Dataset")
):
    import json
    from omegaconf import OmegaConf
    from components.model_training_component import ModelTrainingComponent

    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.create(json.loads(cfg_json))
    ModelTrainingComponent(cfg).execute(
        input_npz_path=input_npz,
        input_model_path=input_model,
        output_model_path=output_model,
        output_history_path=output_history
    )


@dsl.component(base_image=BASE_IMAGE)
def evaluate_model_op(cfg_json: str, input_history: dsl.InputPath("Dataset"), output_plot: dsl.OutputPath("Artifact")):
    import json
    import logging
    from omegaconf import OmegaConf
    from components.model_evaluation_component import ModelEvaluationComponent

    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.create(json.loads(cfg_json))
    ModelEvaluationComponent(cfg).execute(
        input_history_path=input_history,
        output_plot_path=output_plot
    )


# ==========================================
# Pipeline DAG Definition
# ==========================================

class BiometricTrainingPipeline(BasePipeline):
    """
    Defines the structured hardware allocation and sequencing logic for the Biometric execution flow.
    """

    def build_dag(self):
        @dsl.pipeline(
            name="multimodal-biometric-pipeline",
            description="Kubeflow pipeline for Fingerprint and Iris recognition"
        )
        def biometric_pipeline(ingest_cfg: str, preprocess_cfg: str, build_cfg: str, train_cfg: str, evaluate_cfg: str):

            ingest = ingest_op(cfg_json=ingest_cfg).set_cpu_request(
                        str(self.cfg.components.ingestion.compute.cpu_request)).set_memory_request(
                        str(self.cfg.components.ingestion.compute.memory_request))

            preprocess = (preprocess_op(cfg_json=preprocess_cfg, input_parquet=ingest.outputs["output_parquet"]).
                        set_cpu_request(
                        str(self.cfg.components.preprocessing.compute.cpu_request)).set_memory_request(
                        str(self.cfg.components.preprocessing.compute.memory_request)))

            build = build_model_op(cfg_json=build_cfg).set_cpu_request(
                        str(self.cfg.components.building.compute.cpu_request)).set_memory_request(
                        str(self.cfg.components.building.compute.memory_request))

            train = (train_model_op(cfg_json=train_cfg,
                    input_npz=preprocess.outputs["output_npz"],
                    input_model=build.outputs["output_model"]).
                    set_gpu_limit(str(self.cfg.components.training.compute.gpu_limit)).set_cpu_request(
                        str(self.cfg.components.training.compute.cpu_request)).set_memory_request(
                        str(self.cfg.components.training.compute.memory_request)))

            evaluate_model_op(cfg_json=evaluate_cfg, input_history=train.outputs['output_history']).set_cpu_request(
                        str(self.cfg.components.evaluation.compute.cpu_request)).set_memory_request(
                        str(self.cfg.components.evaluation.compute.memory_request))

        return biometric_pipeline


# ==========================================
# Execution & Compilation
# ==========================================

if __name__ == "__main__":
    logger.info("Initializing runtime config parameters from Hydra...")
    cfg = OmegaConf.load("config/config.yaml")

    pipeline = BiometricTrainingPipeline()

    pipeline.compile(
        package_path="biometric_pipeline.yaml",
        pipeline_parameters={
            "ingest_cfg": json.dumps(OmegaConf.to_container(cfg.components.ingestion, resolve=True)),
            "preprocess_cfg": json.dumps(OmegaConf.to_container(cfg.components.preprocessing, resolve=True)),
            "build_cfg": json.dumps(OmegaConf.to_container(cfg.components.building, resolve=True)),
            "train_cfg": json.dumps(OmegaConf.to_container(cfg.components.training, resolve=True)),
            "evaluation_cfg": json.dumps(OmegaConf.to_container(cfg.components.evaluation, resolve=True))
        }
    )
