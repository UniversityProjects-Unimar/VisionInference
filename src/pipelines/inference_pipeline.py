import time
from typing import Iterable, Optional

import numpy as np

from src.config.settings import settings
from src.config.logger import LoggerClass as logger
from src.inference.detector import Detector
from src.inference.schemas import InferenceResult
from src.pipelines.camera_stream import CameraConfig, CameraStream


class InferencePipeline:
    def __init__(self, sources: Optional[Iterable[str]] = None, warmup: bool = True):
        self._sources = list(sources or settings.SOURCES)
        self._detector = Detector()
        if warmup:
            self._detector.warmup()
    
    def run(self):
        for source in self._sources:
            config = CameraConfig(source=source, name=source)
            with CameraStream(config) as stream:
                logger.info(f"Starting inference loop for {stream.name}")
                while True:
                    frame = stream.read()
                    if frame is None:
                        break
                    inference = self._infer(frame, source=stream.name)
                    self._handle_result(inference)
    
    def _infer(self, frame: np.ndarray, source: str) -> InferenceResult:
        return self._detector.predict(frame, source)
    
    def _handle_result(self, result: InferenceResult) -> None:
        logger.info(f"{result.source}: {result.summary()} ({result.latency_ms:.2f}ms)")
        # add api call logic
