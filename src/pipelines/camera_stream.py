from dataclasses import dataclass
from typing import Optional

import cv2

from src.config.logger import LoggerClass as logger

@dataclass(slots=True)
class CameraConfig:
    source: str
    name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture = cv2.VideoCapture(config.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {config.source}")
        self._configure_capture()
        logger.info(f"Camera stream opened: {self.name}")
    
    @property
    def name(self) -> str:
        return self._config.name or self._config.source
    
    def _configure_capture(self) -> None:
        if self._config.width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        if self._config.height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        if self._config.fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, self._config.fps)

    def read(self) -> None:
        ok, frame = self._capture.read()
        if not ok:
            logger.warning(f"Failed to read frame from {self.name}")
            return None
        return frame
    
    def release(self) -> None:
        self._capture.release()
        logger.info(f"Camera stream closed: {self.name}")
      
    def __enter__(self) -> "CameraStream":
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        
