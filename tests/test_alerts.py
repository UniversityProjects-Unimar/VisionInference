"""
Test script for the alert system

This script demonstrates the alert system functionality
and allows testing without a real backend.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from src.config.logger import LoggerClass as logger
from src.inference.schemas import Detection, InferenceResult
from src.utils.alert_manager import AlertManager
from src.utils.video_buffer import VideoBuffer


def test_alert_system():
    """Test the alert system with simulated violations"""
    
    print("=" * 70)
    print("🧪 TESTING ALERT SYSTEM")
    print("=" * 70)
    
    # Create alert manager
    alert_manager = AlertManager(
        violation_duration_threshold=5.0,
        confidence_threshold=0.75,
        backend_url="http://localhost:8080/api/inference"
    )
    
    # Create video buffer
    video_buffer = VideoBuffer(buffer_seconds=10.0, fps=30.0)
    
    # Create dummy frames
    print("\n📹 Creating dummy video frames...")
    frames = []
    for i in range(300):  # 10 seconds at 30fps
        frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        # Add frame number text
        cv2.putText(
            frame,
            f"Frame {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        frames.append(frame)
        video_buffer.add_frame(frame)
    
    print(f"✅ Created {len(frames)} frames")
    print(f"📊 Buffer duration: {video_buffer.get_buffer_duration():.1f}s")
    
    # Simulate violations
    print("\n⚠️  Simulating safety violations...")
    
    # Normal detections (no violation)
    print("\n1️⃣ Sending normal detections (helmet detected)...")
    for i in range(10):
        result = InferenceResult(
            timestamp=time.time(),
            source="test_camera",
            detections=[
                Detection(
                    bbox=(100, 100, 200, 200),
                    class_id=1,
                    class_name="helmet",
                    confidence=0.92
                )
            ],
            latency_ms=45.0
        )
        violation = alert_manager.process_result(result)
        assert violation is None, "Should not trigger violation for helmet"
        time.sleep(0.1)
    print("✅ No violations triggered (correct)")
    
    # Low confidence violation (should not trigger)
    print("\n2️⃣ Sending low confidence violations...")
    for i in range(60):  # 6 seconds
        result = InferenceResult(
            timestamp=time.time(),
            source="test_camera",
            detections=[
                Detection(
                    bbox=(100, 100, 200, 200),
                    class_id=0,
                    class_name="head",
                    confidence=0.65  # Below threshold
                )
            ],
            latency_ms=45.0
        )
        violation = alert_manager.process_result(result)
        assert violation is None, "Should not trigger violation for low confidence"
        time.sleep(0.1)
    print("✅ No violations triggered (correct - confidence too low)")
    
    # High confidence violation (should trigger after 5 seconds)
    print("\n3️⃣ Sending high confidence violations...")
    violation_triggered = False
    for i in range(60):  # 6 seconds
        result = InferenceResult(
            timestamp=time.time(),
            source="test_camera",
            detections=[
                Detection(
                    bbox=(100, 100, 200, 200),
                    class_id=0,
                    class_name="head",
                    confidence=0.89  # Above threshold
                )
            ],
            latency_ms=45.0
        )
        violation = alert_manager.process_result(result)
        
        if violation and not violation_triggered:
            print(f"\n🚨 VIOLATION TRIGGERED!")
            print(f"   ID: {violation.violation_id}")
            print(f"   Duration: {violation.duration_seconds:.1f}s")
            print(f"   Confidence: {violation.max_confidence:.2f}")
            print(f"   Frames: {violation.frame_count}")
            
            # Save test video
            incidents_dir = Path("incidents/test")
            video_path = incidents_dir / f"test_violation_{int(time.time())}.mp4"
            
            print(f"\n💾 Saving incident video...")
            success = video_buffer.save_video(video_path)
            if success:
                print(f"✅ Video saved: {video_path}")
            else:
                print(f"❌ Failed to save video")
            
            # Try to send notification (will fail if backend not running)
            print(f"\n📤 Attempting to send notification...")
            success = alert_manager.send_notification(violation, video_path)
            if success:
                print(f"✅ Notification sent successfully")
            else:
                print(f"⚠️  Notification failed (backend not running?)")
            
            violation_triggered = True
        
        time.sleep(0.1)
    
    assert violation_triggered, "Violation should have been triggered"
    print("\n✅ Violation correctly triggered after 5 seconds")
    
    # Test cooldown
    print("\n4️⃣ Testing cooldown period...")
    for i in range(30):  # 3 seconds
        result = InferenceResult(
            timestamp=time.time(),
            source="test_camera",
            detections=[
                Detection(
                    bbox=(100, 100, 200, 200),
                    class_id=0,
                    class_name="head",
                    confidence=0.89
                )
            ],
            latency_ms=45.0
        )
        violation = alert_manager.process_result(result)
        # Should not trigger due to cooldown
        assert violation is None, "Should not trigger during cooldown"
        time.sleep(0.1)
    print("✅ Cooldown working correctly")
    
    # Statistics
    print("\n📊 Final Statistics:")
    stats = alert_manager.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    logger.configure("alert_test", debug=True)
    test_alert_system()
