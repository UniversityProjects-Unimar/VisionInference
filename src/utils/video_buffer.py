"""Video buffer for storing recent frames to save incident recordings"""

import subprocess
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
        codec: str = 'X264'
    ) -> bool:
        """
        Save buffered frames to video file
        
        Args:
            output_path: Path where to save the video
            fps: Frames per second for output video (defaults to calculated from timestamps)
            codec: Video codec fourcc code (default: 'X264' = H.264)
            
        Returns:
            True if video was saved successfully
        """
        if not self.frames:
            logger.warning("No frames in buffer to save")
            return False
        
        # Calculate actual FPS from timestamps for accurate playback speed
        if fps is None and len(self.timestamps) > 1:
            actual_duration = self.timestamps[-1] - self.timestamps[0]
            if actual_duration > 0:
                fps = len(self.frames) / actual_duration
                logger.debug(f"Calculated FPS from timestamps: {fps:.2f}")
            else:
                fps = self.fps
        elif fps is None:
            fps = self.fps
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get frame dimensions from first frame
        height, width = self.frames[0].shape[:2]
        
        # Try multiple codecs in order of preference
        codecs_to_try = [
            ('X264', 'H.264'),      # Best quality and compatibility
            ('avc1', 'H.264 alt'),  # Alternative H.264
            ('mp4v', 'MPEG-4'),     # Fallback
            ('XVID', 'Xvid'),       # Another fallback
        ]
        
        writer = None
        used_codec = None
        
        for codec_fourcc, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
                test_writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (width, height)
                )
                
                if test_writer.isOpened():
                    writer = test_writer
                    used_codec = codec_name
                    logger.debug(f"Using codec: {codec_name} ({codec_fourcc})")
                    break
                else:
                    test_writer.release()
            except Exception as e:
                logger.debug(f"Codec {codec_fourcc} not available: {e}")
                continue
        
        if writer is None or not writer.isOpened():
            logger.error(f"Failed to open video writer for {output_path} with any codec")
            return False
        
        try:
            # Write all frames
            for frame in self.frames:
                writer.write(frame)
            
            duration = len(self.frames) / fps
            logger.info(f"Saved {len(self.frames)} frames ({duration:.1f}s) to {output_path} using {used_codec}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            return False
            
        finally:
            # CRITICAL: Release writer BEFORE conversion
            writer.release()
            
            # Convert to H.264 if not already using it (only if save was successful)
            if used_codec not in ['H.264', 'H.264 alt']:
                try:
                    logger.info(f"Converting {output_path} to H.264 for better browser compatibility...")
                    h264_success = self._convert_to_h264(output_path)
                    if h264_success:
                        logger.info(f"Successfully converted to H.264")
                    else:
                        logger.warning(f"H.264 conversion failed, keeping original {used_codec} video")
                except Exception as conv_error:
                    logger.error(f"Error during H.264 conversion: {conv_error}")
    
    def clear(self) -> None:
        """Clear all frames from buffer"""
        self.frames.clear()
        self.timestamps.clear()
    
    def get_buffer_duration(self) -> float:
        """Get current duration of buffered video in seconds"""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
    
    def _convert_to_h264(self, video_path: Path) -> bool:
        """
        Convert video to H.264 using ffmpeg for better browser compatibility
        
        Args:
            video_path: Path to video file to convert
            
        Returns:
            True if conversion was successful
        """
        try:
            temp_path = video_path.with_suffix('.tmp.mp4')
            
            # Run ffmpeg conversion - preserve framerate from input
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',  # Ensures compatibility
                '-movflags', '+faststart',  # Enables progressive download
                '-vsync', 'cfr',  # Constant frame rate
                '-y',  # Overwrite output
                str(temp_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and temp_path.exists():
                # Replace original with converted version
                temp_path.replace(video_path)
                logger.debug(f"FFmpeg conversion successful")
                return True
            else:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                if temp_path.exists():
                    temp_path.unlink()
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg conversion timed out after 30s")
            return False
        except Exception as e:
            logger.error(f"Error during H.264 conversion: {e}")
            return False
    
    @property
    def frame_count(self) -> int:
        """Get number of frames currently in buffer"""
        return len(self.frames)
