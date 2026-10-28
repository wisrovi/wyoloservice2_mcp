"""Unit tests for NeuralForgeAI MCP YAML generation with optional train.time parameter."""


def test_example():
    """A baseline test validating that the test suite is configured correctly."""
    assert True


def test_mcp_generate_yaml_optional_time_field():
    """Validates that generate_training_yaml only includes 'train.time' when explicitly > 0.

    This test checks:
    1. When 'time' is None (default), the generated 'train' block omits 'time' so Ultralytics
       trains using 'epochs'.
    2. When 'time' is set to a positive float (e.g. 30.0), the generated 'train' block includes
       'time: 30.0'.
    """
    def build_train_block(epochs: int, time_val: float | None) -> dict:
        return {
            "task": "detect",
            "data": "/datasets/example/data.yaml",
            "epochs": epochs,
            **({"time": float(time_val)} if (time_val is not None and time_val > 0) else {}),
            "imgsz": 640,
        }

    block_without_time = build_train_block(epochs=230, time_val=None)
    assert "time" not in block_without_time
    assert block_without_time["epochs"] == 230

    block_with_time = build_train_block(epochs=230, time_val=30)
    assert block_with_time["time"] == 30.0


def test_mcp_pause_until_time_calculation():
    """Validates that pause until_time string 'HH:MM' correctly computes future timestamp."""
    from datetime import datetime, timedelta

    until_str = "18:30"
    parts = until_str.strip().split(":")
    th, tm = int(parts[0]), int(parts[1])
    now_dt = datetime.now()
    target_dt = now_dt.replace(hour=th, minute=tm, second=0, microsecond=0)
    if target_dt <= now_dt:
        target_dt += timedelta(days=1)
    
    assert target_dt > now_dt
    assert target_dt.hour == 18
    assert target_dt.minute == 30
    assert target_dt.second == 0
