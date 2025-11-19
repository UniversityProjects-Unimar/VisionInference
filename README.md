# VisionInference

Real-time safety helmet detection system using YOLOv11m for industrial environments. Processes video streams from cameras to detect safety equipment compliance.

## Features

- 🎯 Real-time object detection with YOLOv11m
- 📹 Multiple camera source support (USB, RTSP, video files)
- 🐳 Docker containerized deployment
- ⚙️ Configurable confidence and IoU thresholds
- 📊 Structured logging with inference metrics
- 🔥 GPU acceleration support (CUDA)

## Project Structure

```
VisionInference/
├── src/
│   ├── config/          # Configuration and settings
│   ├── inference/       # Model loading and detection logic
│   ├── pipelines/       # Camera streams and inference pipeline
│   └── main.py          # Application entry point
├── models/              # YOLO model weights (.pt files)
├── logs/                # Application logs
├── tests/               # Unit tests
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-container orchestration
└── requirements.txt     # Python dependencies
```

## Setup

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- USB camera or RTSP stream (for live inference)
- CUDA-capable GPU (optional, for faster inference)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TrayFinder/VisionInference.git
   cd VisionInference
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO model**
   ```bash
   mkdir -p models
   # Place your trained yolov11m.pt model in the models/ directory
   # Or download a pretrained model:
   # wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11m.pt -P models/
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

6. **Run locally**
   ```bash
   python -m src.main
   ```

### Docker Deployment

1. **Build the container**
   ```bash
   docker-compose build
   ```

2. **Run the service**
   ```bash
   docker-compose up -d
   ```

3. **View logs**
   ```bash
   docker-compose logs -f vision-inference
   ```

4. **Stop the service**
   ```bash
   docker-compose down
   ```

### USB Camera Setup (Docker)

For USB camera access inside Docker, ensure your `docker-compose.yml` includes device mapping:

```yaml
devices:
  - /dev/video0:/dev/video0
```

On Linux, add your user to the `video` group:
```bash
sudo usermod -aG video $USER
```

## Configuration

Edit `.env` to customize behavior:

## Usage Examples

### Single USB Camera
```properties
SOURCES__0=0
```

### Multiple Cameras
```properties
SOURCES__0=0
SOURCES__1=1
SOURCES__2=rtsp://192.168.1.100:554/stream
```

### Video File
```properties
SOURCES__0=/path/to/safety_video.mp4
```

## Development

### Run Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
isort src/
```

### Type Checking
```bash
mypy src/
```

## Troubleshooting

### Camera not detected in Docker
- Verify device mapping: `ls /dev/video*`
- Check permissions: `ls -l /dev/video0`
- Test locally first: `python -m src.main`

### CUDA out of memory
- Reduce batch size
- Use smaller model (yolov11n or yolov11s)
- Switch to CPU: `DEVICE_PREFERENCE=cpu`

### Model not found
- Ensure model file exists in `models/` directory
- Check `MODEL_PATH` in `.env`
- Verify file permissions

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
