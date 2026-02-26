"""
Interface dictating how pipelines are compiled and structured for Kubernetes environments.
"""
from abc import ABC, abstractmethod
import logging
from kfp import compiler


class BasePipeline(ABC):
    """
    Abstract base class standardizing Kubeflow Pipeline creation.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def build_dag(self, *args, **kwargs):
        """
        Constructs the Directed Acyclic Graph (DAG) logic.
        Must return the KFP @dsl.pipeline decorated function.
        """
        pass

    def compile(self, package_path: str, pipeline_parameters: dict = None) -> None:
        """
        Invokes the KFP compiler to translate the DAG into a deployable YAML manifest.

        Args:
            package_path (str): The destination file path for the YAML.
            pipeline_parameters (dict, optional): Runtime parameters for the pipeline execution.
        """
        pipeline_func = self.build_dag()

        self.logger.info(f"Compiling pipeline manifest to {package_path}...")
        compiler.Compiler().compile(
            pipeline_func=pipeline_func,
            package_path=package_path,
            pipeline_parameters=pipeline_parameters or {}
        )
        self.logger.info("Kubeflow YAML compilation completed successfully.")