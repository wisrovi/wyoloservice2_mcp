"""Data Transfer Objects for NeuralForgeAI MCP Server."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    """Configuration container for NeuralForge training studies.

    Attributes:
        name: Name of the study. MUST follow the format <project_name>_<dataset_name>.
        dataset: Absolute path to the dataset.yaml file.
        task: Task type ('detect', 'segment', 'classify', 'obb', 'pose').
        epochs: Number of epochs.
        time: Optional max training time per trial in hours.
        models: List of model weights to evaluate.
        imgsz: List of image resolutions to search.
        n_trials: Number of hyperparameter optimization trials to run.
        enable_xai: Enable post-train Explainability (Eigen-CAM / Grad-CAM / Deletion AUC).
        enable_adversarial: Enable post-train adversarial robustness testing.
        enable_latex: Enable PyLaTeX export for scientific tables and reports.
        sampler: Optuna hyperparameter optimization sampler ('tpe' or 'cmaes').
        metadata_content: Short description of dataset content.
        metadata_documentation: Detailed documentation about dataset classes.
        target_worker: Optional target worker name or IP for private queue routing.
        start_at: Optional scheduled execution time string.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Name of the study. MUST follow the format <project_name>_<dataset_name>.")
    dataset: str = Field(..., description="Absolute path to the dataset.yaml file")
    task: str = Field(..., description="Task type: 'detect', 'segment', 'classify', 'obb', or 'pose'")
    epochs: int = Field(100, description="Number of epochs")
    time: Optional[float] = Field(default=None, description="Optional max training time per trial in hours.")
    models: List[str] = Field(default=["yolo26n.pt"], description="List of models to try")
    imgsz: List[int] = Field(default=[640], description="List of image sizes (imgsz) to try")
    n_trials: int = Field(default=3, description="Number of hyperparameter optimization trials to run")
    enable_xai: bool = Field(default=True, description="Enable post-train Explainability")
    enable_adversarial: bool = Field(default=True, description="Enable post-train adversarial robustness testing")
    enable_latex: bool = Field(default=True, description="Enable PyLaTeX export")
    sampler: str = Field(default="tpe", description="Optuna sampler: 'tpe' or 'cmaes'")
    metadata_content: str = Field(default="YOLO training dataset", description="Dataset description")
    metadata_documentation: str = Field(default="Auto-generated training study", description="Dataset documentation")
    target_worker: Optional[str] = Field(default=None, description="Optional target worker name or IP")
    start_at: Optional[str] = Field(default=None, description="Optional scheduled execution time")
