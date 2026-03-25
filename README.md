# Tennis Video Tracker

Real-time tennis video analysis with player outlines, court line detection, ball tracking, and speed measurement.

## Features
- Player bounding boxes + full pose skeleton overlay (MediaPipe)
- Court line detection (Hough transform)
- Ball detection + motion trail (YOLOv8)
- Real-time ball speed (km/h and mph) via Kalman-filtered tracking

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Run

```bash
# On a video file
python src/main.py --input path/to/match.mp4

# On webcam
python src/main.py --input 0

# Save annotated output
python src/main.py --input match.mp4 --output annotated.mp4
```

### Controls
| Key | Action |
|-----|--------|
| `q` | Quit |
| `Space` | Pause / Resume |
| `s` | Save screenshot |

### Flags
| Flag | Description |
|------|-------------|
| `--no-skeleton` | Hide pose skeleton |
| `--no-court` | Hide court lines |
| `--no-speed` | Hide speed HUD |

## Project Structure

```
src/
├── detection/
│   ├── ball_detector.py      # YOLOv8 ball detection + trail
│   ├── player_detector.py    # YOLOv8 person + MediaPipe pose
│   └── court_detector.py     # Hough line detection + homography calibration
├── tracking/
│   └── speed_calculator.py   # Kalman filter + real-world speed
├── overlay/
│   └── renderer.py           # All visual overlays
└── main.py                   # Pipeline entry point
models/                        # Place .pt weight files here
```

## Improving Accuracy

- **Ball detection**: Replace `yolov8n.pt` with a tennis-specific model like [TrackNetV3](https://github.com/Chang-Chia-Chi/TrackNet) for much better ball tracking at high speeds.
- **Speed accuracy**: Call `court_det.calibrate(frame, corners)` with the 4 court baseline corners to enable real-world metre conversion.
- **GPU acceleration**: Install `torch` with CUDA support for real-time performance on NVIDIA GPUs.
