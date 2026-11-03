"""Unit tests for newly added NeuralForgeAI MCP tools and configurations."""
import pytest
from wyolo_mcp.server import (
    TrainingConfig,
    generate_training_yaml,
)


def test_training_config_yolo26_defaults():
    """Validates that TrainingConfig defaults to yolo26 and includes XAI toggles."""
    config = TrainingConfig(
        name="project_dataset",
        dataset="/data/datasets/coco.yaml",
        task="obb"
    )
    assert config.models == ["yolo26n.pt"]
    assert config.enable_xai is True
    assert config.enable_adversarial is True
    assert config.enable_latex is True
    assert config.sampler == "tpe"
    assert config.task == "obb"


def test_generate_training_yaml_with_post_train(tmp_path):
    """Validates generate_training_yaml outputs post_train block and fitness metric for obb."""
    config = TrainingConfig(
        name="test_obb_study",
        dataset="/data/datasets/obb.yaml",
        task="obb",
        models=["yolo26s.pt"],
        epochs=50,
        enable_xai=True,
        enable_adversarial=True,
        enable_latex=True,
        sampler="cmaes"
    )

    res = generate_training_yaml(config, output_dir=str(tmp_path))
    assert res.get("success") is True
    yaml_path = res["yaml_path"]

    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["model"] == "yolo26s.pt"
    assert data["train"]["task"] == "obb"
    assert "post_train" in data
    assert data["post_train"]["xai"] is True
    assert data["post_train"]["adversarial"] is True
    assert data["post_train"]["latex"] is True
    assert data["sweeper"]["sampler"] == "cmaes"
    assert data["sweeper"]["fitness"] == "metrics/mAP50-95(B)"


@pytest.mark.asyncio
async def test_estimate_duration_heuristic():
    """Validates heuristic fallback calculations for estimate_study_duration."""
    from wyolo_mcp.server import estimate_study_duration
    
    # Run estimate without live K8s (falls back to heuristic)
    res = await estimate_study_duration(dataset_path="/data/test.yaml", epochs=100, model="yolo26n.pt")
    assert res["status"] == "success"
    assert res["estimated_minutes"] > 0
    assert res["estimated_hours"] > 0
    assert res["parameters"]["model"] == "yolo26n.pt"
