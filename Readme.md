# Multimodal Biometric Recognition System (MLOps)

A distributed, production-grade machine learning pipeline for multimodal biometric recognition (Fingerprint + Iris). This repository implements a strict Object-Oriented architecture, replacing in-memory data passing with a highly optimized, artifact-based Directed Acyclic Graph (DAG) for execution on Kubeflow or AWS SageMaker Pipelines.

The pipeline emits standardized MLflow telemetry and JSON artifacts at every run, specifically optimized for seamless ingestion by multi-cloud, multi-account monitoring dashboards spanning AWS, Azure, and GCP.

## 🏗 Pipeline Architecture

The pipeline executes in strictly isolated Kubernetes Pods. Every component inherits from a central `BaseComponent` interface and relies on file-based artifact handshakes:

1. **Ingestion (`DataIngestionComponent`):** Uses Ray to distribute disk I/O, packaging raw fingerprint and iris imagery into a columnar PyArrow Parquet artifact.
2. **Preprocessing (`DataPreprocessingComponent`):** Ingests Parquet, applies tensor normalization and categorical encoding, and exports an `.npz` artifact.
3. **Model Building (`ModelBuildingComponent`):** Compiles a multi-branch Keras network (MobileNetV2 backbone + custom Iris CNNs) and saves the untrained `.h5` artifact.
4. **Model Training (`ModelTrainingComponent`):** Executes GPU-accelerated training. Automatically streams parameters, epoch metrics, and model weights to a centralized MLflow server, outputting the finalized `.h5` and a `history.json`.
5. **Evaluation (`ModelEvaluationComponent`):** Parses the JSON telemetry to generate evaluation plots and visualizations.

## 📁 Repository Structure

```text
multimodal-biometric-system/
├── .github/workflows/        # CI/CD (Lint, Pytest, Docker Build, KFP Compile)
├── build/                    # Contains the Dockerfile for the GPU execution environment
├── components/               # Object-oriented pipeline components and base interfaces
├── config/                   # Centralized Hydra configurations (config.yaml)
├── pipelines/                # Pipeline's DAG definition and compiler
├── requirements/             # Pinned Python dependencies
├── tests/                    # Pytest suite utilizing mocked dependencies and temp files
├── utils/                    # Global utilities (e.g., Reproducibility seed locks)
└── README.md                 # System documentation