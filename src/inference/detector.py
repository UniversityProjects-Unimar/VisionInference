from pyexpat import model
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from src.config.settings import settings
from src.config.logger import LoggerClass as logger
from src.inference.model_loader import DeviceType, ModelInfo, ModelLoader
from src.inference.schemas import Detection, InferenceResult

class Detector:
    def __init__(
            self,
            model_path: Optional[str] = None,
            confidence: Optional[str] = None,
            iou: Optional[str] = None
    ) -> None:
        self._loader = ModelLoader()
        self._model_path = model_path or str(settings.DEFAULT_MODEL_PATH)
        self._confidence = confidence or settings.DETECTION_CONFIDENCE_THRESHOLD
        self._iou = iou or settings.DETECTION_IOU
        self._model, self._info = self._loader.load_model(self._model_path)
    
    @property
    def model_info(self) -> ModelInfo:
        return self._info
    
    def predict(self, frame: np.ndarray, source: str) -> InferenceResult:
        start = time.perf_counter()
        results = self._model.predict(
            frame,
            imgsz=self._info.input_size,
            conf=self._confidence,
            iou=self._iou,
            verbose=False
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        detections = self._convert_results(results[0])
        logger.debug(f"Inference finished in {latency_ms:.2f} ms with {len(detections)} detections.")
        return InferenceResult(
            timestamp=datetime.now(timezone.utc),
            source=source,
            detections=detections,
            latency_ms=latency_ms
        )

    def warmup(self, runs: int = settings.WARMUP_RUNS) -> None:
        if runs <= 0:
            return
        
        logger.info(f"Warming up the model with {runs} dummy runs...")
        dummy = np.zeros((self._info.input_size, self._info.input_size, 3), dtype=np.uint8)
        for _ in range(runs):
            self._model.predict(
                dummy,
                imgsz=self._info.input_size,
                conf=self._confidence,
                iou=self._iou,
                verbose=False
            )
    

    def _convert_results(self, results) -> List[Detection]:
        detections: List[Detection] = []
        boxes = getattr(results, "boxes", [])
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf.item()) if hasattr(box, "conf") else 0.0
            cls_id = int(box.cls.item()) if hasattr(box, "cls") else -1
            class_name = self._info.class_names.get(cls_id, str(cls_id))
            detections.append(Detection(
                bbox=tuple(xyxy),
                confidence=conf,
                class_id=cls_id,
                class_name=class_name
            ))
        return detections
