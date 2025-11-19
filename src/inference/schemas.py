from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Dict, List, Tuple
from datetime import datetime


class DeviceType(StrEnum):
    """Device type enumeration"""
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

@dataclass(slots=True)
class Detection:
    """Stores information about a single detection"""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float


@dataclass(slots=True)
class InferenceResult:
    timestamp: datetime
    source: str
    detections: List[Detection]
    latency_ms: float

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts