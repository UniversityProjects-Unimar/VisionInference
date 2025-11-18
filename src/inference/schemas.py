from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Dict

class DeviceType(StrEnum):
    CPU = auto()
    CUDA = auto()

@dataclass
class ModelInfo:
    """Stores metadata about a loaded model"""

    model_path: str
    model_name: str
    input_size: int # e.g., 640
    num_classes: int
    class_names: Dict[int, str]
    device: DeviceType

    def to_dict(self) -> Dict:
        """Convert ModelInfo to dictionary"""
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "device": self.device.value,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }

