"""Alert manager for tracking safety violations and sending notifications"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

from src.config.logger import LoggerClass as logger
from src.config.settings import settings
from src.inference.schemas import InferenceResult


@dataclass
class SafetyViolation:
    """Represents a detected safety violation"""
    
    violation_id: str
    source: str
    violation_type: str  # e.g., "no_helmet"
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    max_confidence: float = 0.0
    frame_count: int = 0
    video_path: Optional[str] = None
    notified: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API payload"""
        return {
            "violation_id": self.violation_id,
            "source": self.source,
            "violation_type": self.violation_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "max_confidence": self.max_confidence,
            "frame_count": self.frame_count,
            "video_path": self.video_path,
        }


class AlertManager:
    """
    Manages safety violation detection and notification.
    
    Tracks consecutive detections of unsafe classes (head, person_no_helmet)
    and triggers alerts when violations exceed duration threshold.
    """
    
    # Classes that indicate safety violations
    VIOLATION_CLASSES: Set[str] = {"head", "person_no_helmet"}
    
    def __init__(
        self,
        violation_duration_threshold: float = 5.0,
        confidence_threshold: float = 0.75,
        backend_url: Optional[str] = None,
        cooldown_seconds: float = 30.0
    ):
        """
        Initialize alert manager
        
        Args:
            violation_duration_threshold: Seconds of consecutive violation before alerting
            confidence_threshold: Minimum confidence to count as violation
            backend_url: Backend API endpoint for notifications
            cooldown_seconds: Minimum time between alerts for same source
        """
        self.violation_duration_threshold = violation_duration_threshold
        self.confidence_threshold = confidence_threshold
        self.backend_url = backend_url or settings.BACKEND_API_URL
        self.cooldown_seconds = cooldown_seconds
        
        # Track active violations per source
        self.active_violations: Dict[str, SafetyViolation] = {}
        
        # Track violation start times per source
        self.violation_start_times: Dict[str, float] = {}
        
        # Track last alert time per source (for cooldown)
        self.last_alert_times: Dict[str, float] = {}
        
        # All violations history
        self.violation_history: List[SafetyViolation] = []
        
        logger.info(f"AlertManager initialized: threshold={violation_duration_threshold}s, confidence={confidence_threshold}, backend={self.backend_url}")
    
    def process_result(self, result: InferenceResult) -> Optional[SafetyViolation]:
        """
        Process inference result and check for violations
        
        Args:
            result: Inference result to process
            
        Returns:
            SafetyViolation if alert should be triggered, None otherwise
        """
        source = result.source
        current_time = time.time()
        
        # Check if any violation class was detected with sufficient confidence
        violation_detected = any(
            det.class_name in self.VIOLATION_CLASSES and 
            det.confidence >= self.confidence_threshold
            for det in result.detections
        )
        
        if violation_detected:
            # Get max confidence from violation detections
            max_conf = max(
                (det.confidence for det in result.detections 
                 if det.class_name in self.VIOLATION_CLASSES),
                default=0.0
            )
            
            # Start or continue tracking violation
            if source not in self.violation_start_times:
                self.violation_start_times[source] = current_time
                logger.debug(f"Violation tracking started for {source}")
            
            violation_duration = current_time - self.violation_start_times[source]
            
            # Update or create active violation
            if source not in self.active_violations:
                violation_id = f"{source}_{int(current_time)}"
                self.active_violations[source] = SafetyViolation(
                    violation_id=violation_id,
                    source=source,
                    violation_type="no_helmet",
                    start_time=datetime.fromtimestamp(
                        self.violation_start_times[source],
                        tz=timezone.utc
                    ),
                    max_confidence=max_conf,
                    frame_count=1
                )
            else:
                # Update existing violation
                violation = self.active_violations[source]
                violation.frame_count += 1
                violation.max_confidence = max(violation.max_confidence, max_conf)
                violation.duration_seconds = violation_duration
            
            # Check if violation exceeded threshold and cooldown passed
            if violation_duration >= self.violation_duration_threshold:
                if self._can_send_alert(source, current_time):
                    violation = self.active_violations[source]
                    violation.end_time = datetime.now(timezone.utc)
                    logger.warning(
                        f"SAFETY VIOLATION: {source} - {violation.violation_type} "
                        f"for {violation_duration:.1f}s (confidence: {max_conf:.2f})"
                    )
                    
                    # Reset tracking for this source after triggering alert
                    # This prevents immediate re-triggering during cooldown
                    del self.violation_start_times[source]
                    del self.active_violations[source]
                    
                    return violation
        
        else:
            # No violation detected - reset tracking
            if source in self.violation_start_times:
                duration = current_time - self.violation_start_times[source]
                logger.debug(f"Violation ended for {source} after {duration:.1f}s")
                del self.violation_start_times[source]
            
            if source in self.active_violations:
                del self.active_violations[source]
        
        return None
    
    def _can_send_alert(self, source: str, current_time: float) -> bool:
        """Check if alert can be sent (respects cooldown)"""
        if source not in self.last_alert_times:
            return True
        
        time_since_last = current_time - self.last_alert_times[source]
        return time_since_last >= self.cooldown_seconds
    
    def send_notification(
        self,
        violation: SafetyViolation,
        video_path: Optional[Path] = None,
        timeout: float = 10.0
    ) -> bool:
        """
        Send notification to backend API as multipart/form-data
        
        Args:
            violation: Violation to report
            video_path: Path to incident video file
            timeout: Request timeout in seconds
            
        Returns:
            True if notification was sent successfully
        """
        if violation.notified:
            logger.debug(f"Violation {violation.violation_id} already notified")
            return True
        
        # Video file is required by backend
        if not video_path or not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        # Update video path
        violation.video_path = str(video_path)
        
        # Prepare multipart form data
        data = {
            'local': f"Camera {violation.source}",
            'category': 'FALTA_DE_EPI'
        }
        
        try:
            logger.info(f"Sending notification to {self.backend_url}")
            
            # Open and send video file
            with open(video_path, 'rb') as video_file:
                files = {
                    'file': (Path(video_path).name, video_file, 'video/mp4')
                }
                
                response = requests.post(
                    self.backend_url,
                    data=data,
                    files=files,
                    timeout=timeout
                )
            
            if response.status_code in (200, 201, 202):
                logger.info(f"Notification sent successfully: {violation.violation_id}")
                violation.notified = True
                self.last_alert_times[violation.source] = time.time()
                self.violation_history.append(violation)
                return True
            else:
                logger.error(
                    f"Failed to send notification: {response.status_code} - {response.text}"
                )
                return False
                
        except requests.exceptions.Timeout:
            logger.error(f"Notification timeout after {timeout}s")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send notification: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending notification: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        return {
            "total_violations": len(self.violation_history),
            "active_violations": len(self.active_violations),
            "sources_tracked": list(self.violation_start_times.keys()),
            "violation_classes": list(self.VIOLATION_CLASSES),
            "threshold_seconds": self.violation_duration_threshold,
            "confidence_threshold": self.confidence_threshold,
        }
    
    def reset(self) -> None:
        """Reset all tracking state"""
        self.active_violations.clear()
        self.violation_start_times.clear()
        logger.info("AlertManager state reset")
