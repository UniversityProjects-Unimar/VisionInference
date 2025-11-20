"""
Inference Test Script

This script tests the inference pipeline with live visualization,
class analysis, and confidence score evaluation.

Usage:
    python inference_test.py
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.detector import Detector
from src.inference.schemas import Detection


class InferenceVisualizer:
    """Visualize and analyze inference results"""

    def __init__(self, show_video: bool = True, save_video: bool = False, device: str = None):
        self.show_video = show_video
        self.save_video = save_video
        
        # Create detector with specified device
        self.detector = Detector(device=device)
        
        self.stats = defaultdict(lambda: {"count": 0, "confidences": []})
        self.frame_count = 0
        self.total_latency = 0.0
        self.video_writer = None

        # Color map for different classes
        self.colors = {
            "helmet": (0, 255, 0),      # Green
            "head": (0, 0, 255),        # Red
            "person": (255, 0, 0),      # Blue
            "vest": (0, 255, 255),      # Yellow
            "no-helmet": (0, 0, 255),   # Red
        }

        print("=" * 70)
        print("🎯 INFERENCE TEST - Safety Helmet Detection")
        print("=" * 70)
        print(f"Model: {self.detector.model_info.model_name}")
        print(f"Device: {self.detector.model_info.device.value.upper()}")
        print(f"Classes: {list(self.detector.model_info.class_names.values())}")
        print(f"Input size: {self.detector.model_info.input_size}px")
        print("=" * 70)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'p' to pause/resume")
        print("  - Press 'r' to reset statistics")
        print("=" * 70)

    def get_color(self, class_name: str) -> tuple:
        """Get color for a class"""
        return self.colors.get(class_name.lower(), (255, 255, 255))

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = self.get_color(det.class_name)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return annotated

    def draw_stats(self, frame: np.ndarray, latency_ms: float) -> np.ndarray:
        """Draw statistics overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent background for stats
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw text
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        avg_fps = 1000.0 / (self.total_latency / self.frame_count) if self.frame_count > 0 else 0

        texts = [
            f"Frame: {self.frame_count}",
            f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})",
            f"Latency: {latency_ms:.1f}ms",
            f"Device: {self.detector.model_info.device.value.upper()}",
        ]

        y_offset = 35
        for text in texts:
            cv2.putText(
                frame,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 25

        return frame

    def update_stats(self, detections: list[Detection]):
        """Update detection statistics"""
        for det in detections:
            self.stats[det.class_name]["count"] += 1
            self.stats[det.class_name]["confidences"].append(det.confidence)

    def print_stats(self):
        """Print detection statistics"""
        print("\n" + "=" * 70)
        print("📊 DETECTION STATISTICS")
        print("=" * 70)

        if not self.stats:
            print("No detections recorded yet.")
            return

        # Sort by count
        sorted_stats = sorted(self.stats.items(), key=lambda x: x[1]["count"], reverse=True)

        print(f"{'Class':<20} {'Count':<10} {'Min Conf':<12} {'Avg Conf':<12} {'Max Conf':<12}")
        print("-" * 70)

        for class_name, data in sorted_stats:
            count = data["count"]
            confidences = data["confidences"]
            min_conf = min(confidences)
            avg_conf = sum(confidences) / len(confidences)
            max_conf = max(confidences)

            print(f"{class_name:<20} {count:<10} {min_conf:<12.3f} {avg_conf:<12.3f} {max_conf:<12.3f}")

        print("=" * 70)
        print(f"\nTotal frames processed: {self.frame_count}")
        print(f"Average FPS: {1000.0 / (self.total_latency / self.frame_count):.2f}")
        print(f"Average latency: {self.total_latency / self.frame_count:.2f}ms")

    def reset_stats(self):
        """Reset statistics"""
        self.stats.clear()
        self.frame_count = 0
        self.total_latency = 0.0
        print("\n✅ Statistics reset!")

    def save_frame(self, frame: np.ndarray):
        """Save current frame"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"inference_test_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"\n✅ Frame saved: {filename}")

    def run(self, source: str = "0"):
        """Run inference test"""
        # Convert string to int for camera index
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Failed to open video source: {source}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n📹 Video source opened: {source}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print("\nStarting inference...\n")

        # Setup video writer if needed
        if self.save_video:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"inference_test_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            print(f"💾 Recording to: {output_file}")

        paused = False

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Failed to read frame. Exiting...")
                        break

                    # Run inference
                    result = self.detector.predict(frame, source=str(source))

                    # Update statistics
                    self.frame_count += 1
                    self.total_latency += result.latency_ms
                    self.update_stats(result.detections)

                    # Draw detections
                    annotated = self.draw_detections(frame, result.detections)
                    annotated = self.draw_stats(annotated, result.latency_ms)

                    # Save frame to video
                    if self.video_writer:
                        self.video_writer.write(annotated)

                    # Show frame
                    if self.show_video:
                        cv2.imshow("Inference Test", annotated)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n🛑 Quit requested...")
                    break
                elif key == ord('s'):
                    self.save_frame(annotated)
                elif key == ord('p'):
                    paused = not paused
                    status = "⏸️  PAUSED" if paused else "▶️  RESUMED"
                    print(f"\n{status}")
                elif key == ord('r'):
                    self.reset_stats()

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user...")

        finally:
            # Cleanup
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            if self.show_video:
                cv2.destroyAllWindows()

            # Print final statistics
            self.print_stats()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test inference with visualization")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (camera index, file path, or RTSP URL)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display (useful for headless systems)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output video to file"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Force specific device (cpu or cuda). Default: auto-detect"
    )

    args = parser.parse_args()

    visualizer = InferenceVisualizer(
        show_video=not args.no_display,
        save_video=args.save_video,
        device=args.device
    )
    visualizer.run(source=args.source)


if __name__ == "__main__":
    main()
