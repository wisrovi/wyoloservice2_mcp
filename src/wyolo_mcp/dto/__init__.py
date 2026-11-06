"""DTO package export for wyolo_mcp package."""

import sys
import os

root_dto = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "dto"))
if root_dto not in sys.path:
    sys.path.insert(0, root_dto)

from training_config_dto import TrainingConfig

__all__ = ["TrainingConfig"]
