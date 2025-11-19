"""Video buffer for storing recent frames to save incident recordings"""

import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config.logger import LoggerClass as logger


class VideoBuffer:
    """
    Ring buffer that stores recent frames for incident recording.
    
    When a safety violation is detected, we can save the last N seconds
    of video showing what happened before and during the incident.
    """

    def __init__(self, buffer_seconds: float = 10.0, fps: float = 30.0):
        """
        Initialize video buffer
        
        Args:
            buffer_seconds: How many seconds of video to keep in buffer
            fps: Expected frames per second
        """
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_frames = int(buffer_seconds * fps)
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        
    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer"""
        self.frames.append(frame.copy())
        self.timestamps.append(time.time())
    
    def save_video(
        self,
        output_path: Path,
        fps: Optional[float] = None,
        codec: str = 'mp4v'
    ) -> bool:
        """
        Save buffered frames to video file
        
        Args:
            output_path: Path where to save the video
            fps: Frames per second for output video (defaults to buffer fps)
            codec: Video codec fourcc code
            
        Returns:
            True if video was saved successfully
        """
        if not self.frames:
            logger.warning("No frames in buffer to save")
            return False
        
        fps = fps or self.fps
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get frame dimensions from first frame
        height, width = self.frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not writer.isOpened():
            logger.error(f"Failed to open video writer for {output_path}")
            return False
        
        try:
            # Write all frames
            for frame in self.frames:
                writer.write(frame)
            
            duration = len(self.frames) / fps
            logger.info(f"Saved {len(self.frames)} frames ({duration:.1f}s) to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            return False
            
        finally:
            writer.release()
    
    def clear(self) -> None:
        """Clear all frames from buffer"""
        self.frames.clear()
        self.timestamps.clear()
    
    def get_buffer_duration(self) -> float:
        """Get current duration of buffered video in seconds"""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
    
    @property
    def frame_count(self) -> int:
        """Get number of frames currently in buffer"""
        return len(self.frames)
