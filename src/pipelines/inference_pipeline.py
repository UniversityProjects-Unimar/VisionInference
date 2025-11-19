import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import requests

from src.config.logger import LoggerClass as logger
from src.config.settings import settings
from src.inference.detector import Detector
from src.inference.schemas import InferenceResult
from src.pipelines.camera_stream import CameraConfig, CameraStream
from src.utils.alert_manager import AlertManager
from src.utils.video_buffer import VideoBuffer


class InferencePipeline:
    def __init__(
        self,
        sources: Optional[Iterable[str]] = None,
        warmup: bool = True,
        enable_alerts: bool = True,
        violation_threshold_seconds: float = 5.0,
        violation_confidence: float = 0.75,
        video_buffer_seconds: float = 10.0,
        incidents_dir: str = "incidents"
    ):
        self._sources = list(sources or settings.SOURCES)
        self._detector = Detector()
        
        # Alert and video recording setup
        self._enable_alerts = enable_alerts
        self._alert_manager = AlertManager(
            violation_duration_threshold=violation_threshold_seconds,
            confidence_threshold=violation_confidence
        ) if enable_alerts else None
        
        # Video buffers per source
        self._video_buffers: Dict[str, VideoBuffer] = {}
        self._video_buffer_seconds = video_buffer_seconds
        self._incidents_dir = Path(incidents_dir)
        self._incidents_dir.mkdir(parents=True, exist_ok=True)
        
        if warmup:
            self._detector.warmup()
    
    def run(self):
        for source in self._sources:
            config = CameraConfig(source=source, name=source)
            
            # Initialize video buffer for this source
            if self._enable_alerts and source not in self._video_buffers:
                # Get FPS from camera or use default
                self._video_buffers[source] = VideoBuffer(
                    buffer_seconds=self._video_buffer_seconds,
                    fps=30.0  # Default, will be updated if available
                )
            
            with CameraStream(config) as stream:
                logger.info(f"Starting inference loop for {stream.name}")
                
                while True:
                    frame = stream.read()
                    if frame is None:
                        break
                    
                    # Add frame to video buffer
                    if self._enable_alerts and source in self._video_buffers:
                        self._video_buffers[source].add_frame(frame)
                    
                    # Run inference
                    inference = self._infer(frame, source=stream.name)
                    
                    # Handle result and check for violations
                    self._handle_result(inference, frame)
    
    def _infer(self, frame: np.ndarray, source: str) -> InferenceResult:
        return self._detector.predict(frame, source)
    
    def _handle_result(self, result: InferenceResult, frame: np.ndarray) -> None:
        """Handle inference result, check for violations, and send notifications"""
        logger.info(f"{result.source}: {result.summary()} ({result.latency_ms:.2f}ms)")
        
        # Check for safety violations if alerts are enabled
        if self._enable_alerts and self._alert_manager:
            violation = self._alert_manager.process_result(result)
            
            if violation:
                # Safety violation detected! Save video and notify
                self._handle_violation(violation, result.source)
        
        # Send regular inference results to backend (optional)
        self._send_inference_result(result)
    
    def _handle_violation(self, violation, source: str) -> None:
        """Handle detected safety violation"""
        logger.warning(f"Handling safety violation: {violation.violation_id}")
        
        # Save incident video
        video_path = None
        if source in self._video_buffers:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_filename = f"violation_{source}_{timestamp}.mp4"
            video_path = self._incidents_dir / video_filename
            
            success = self._video_buffers[source].save_video(video_path)
            if success:
                logger.info(f"Incident video saved: {video_path}")
            else:
                logger.error(f"Failed to save incident video for {violation.violation_id}")
        
        # Send notification to backend
        if self._alert_manager:
            self._alert_manager.send_notification(violation, video_path)
    
    def _send_inference_result(self, result: InferenceResult) -> None:
        """Send regular inference result to backend (optional)"""
        try:
            response = requests.post(
                "http://localhost:8000/api/inference",
                json={
                    "timestamp": result.timestamp.isoformat(),
                    "source": result.source,
                    "detections": [
                        {
                            "bbox": det.bbox,
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "confidence": det.confidence,
                        }
                        for det in result.detections
                    ],
                    "latency_ms": result.latency_ms,
                },
                timeout=5,
            )
            response.raise_for_status()
            logger.debug(f"Successfully sent inference result for {result.source}")
        except requests.RequestException as e:
            logger.debug(f"Failed to send inference result for {result.source}: {e}")