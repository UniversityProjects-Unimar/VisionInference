from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",  # Support nested env variables
        extra="ignore",
    )

    PROJECT_NAME: str = "VisionInference"
    LOGS_DIR: Path = Path("logs")
    MODELS_DIR: Path = Path("models")
    DEFAULT_MODEL_PATH: Path = Field(default="models/yolov11m.pt", alias="MODEL_PATH")
    DETECTION_CONFIDENCE_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)
    DETECTION_IOU: float = Field(default=0.45, ge=0.0, le=1.0)
    DEVICE_PREFERENCE: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
    SOURCES: List[str] = Field(default_factory=list)
    WARMUP_RUNS: int = Field(default=2, ge=0, le=10) # for handling initial setup tasks

    def ensure_directories(self):
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_app_settings() -> AppSettings:
    cfg = AppSettings()
    cfg.ensure_directories()
    return cfg


settings = get_app_settings()