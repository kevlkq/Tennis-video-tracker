import cv2
import numpy as np
from collections import deque


class BallDetector:
    """
    Detects and tracks the tennis ball.

    Supports two backends:
      - 'yolo'       : YOLOv8 (fast, works out of the box, weaker on fast balls)
      - 'tracknetv3' : TrackNetV3 (heatmap-based, much better on fast/blurry balls)

    Usage:
        # YOLO (default, no extra setup)
        detector = BallDetector(backend='yolo', model_path='yolov8n.pt')

        # TrackNetV3 (download weights first from the TrackNetV3 GitHub repo)
        detector = BallDetector(backend='tracknetv3', model_path='models/tracknetv3.pt')
    """

    def __init__(
        self,
        backend: str = "yolo",
        model_path: str = "yolov8n.pt",
        confidence: float = 0.3,
        trail_length: int = 20,
    ):
        self.backend = backend.lower()
        self.confidence = confidence
        self.trail: deque = deque(maxlen=trail_length)
        self._frame_buffer: deque = deque(maxlen=3)  # TrackNet needs 3 frames

        if self.backend == "yolo":
            self._load_yolo(model_path)
        elif self.backend == "tracknetv3":
            self._load_tracknetv3(model_path)
        elif self.backend == "artlabss":
            self._load_artlabss(model_path)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'yolo', 'tracknetv3', or 'artlabss'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        Run detection on a single frame.
        Returns dict: {bbox, center, confidence} or None if no ball found.
        """
        self._frame_buffer.append(frame)

        if self.backend == "yolo":
            return self._detect_yolo(frame)
        elif self.backend == "artlabss":
            return self._detect_artlabss()
        else:
            return self._detect_tracknetv3()

    def get_trail(self) -> list:
        return list(self.trail)

    # ------------------------------------------------------------------
    # YOLO backend
    # ------------------------------------------------------------------

    def _load_yolo(self, model_path: str) -> None:
        from ultralytics import YOLO
        self._model = YOLO(model_path)

    def _detect_yolo(self, frame: np.ndarray) -> dict | None:
        results = self._model(frame, verbose=False, conf=self.confidence, classes=[32])
        if not results or len(results[0].boxes) == 0:
            self.trail.append(None)
            return None

        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        box = boxes[best_idx]

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        conf = float(box.conf[0])

        self.trail.append((cx, cy))
        return {"bbox": (x1, y1, x2, y2), "center": (cx, cy), "confidence": conf}

    # ------------------------------------------------------------------
    # TrackNetV3 backend
    # ------------------------------------------------------------------

    def _load_tracknetv3(self, model_path: str) -> None:
        """
        Load TrackNetV3 weights.

        TrackNetV3 setup:
          1. Clone the TrackNetV3 repo from GitHub (search 'TrackNetV3 tennis')
          2. Download pretrained weights (.pt file) from the repo's releases
          3. Place the .pt file in the models/ folder
          4. pip install torch torchvision

        The model expects 3 stacked BGR frames resized to 512x288,
        and outputs a heatmap of the same size.
        """
        import torch
        import sys
        import os

        # The TrackNetV3 repo must be cloned alongside this project
        # or its path added here
        tracknet_path = os.path.join(os.path.dirname(__file__), "..", "..", "TrackNetV3")
        if os.path.exists(tracknet_path):
            sys.path.insert(0, tracknet_path)

        try:
            from model import TrackNet  # TrackNetV3 repo exposes this
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tracknet = TrackNet(in_channels=9, out_channels=3)  # 3 frames × 3 channels
            state = torch.load(model_path, map_location=self._device)
            self._tracknet.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
            self._tracknet.to(self._device).eval()
            print(f"[TrackNetV3] Loaded weights from {model_path} on {self._device}")
        except ImportError:
            raise ImportError(
                "TrackNetV3 model class not found. "
                "Clone the TrackNetV3 repo into the project root as 'TrackNetV3/' first."
            )

    def _detect_tracknetv3(self) -> dict | None:
        """Run TrackNetV3 inference on the last 3 buffered frames."""
        import torch

        if len(self._frame_buffer) < 3:
            self.trail.append(None)
            return None

        INPUT_W, INPUT_H = 512, 288

        # Stack 3 frames → (9, H, W) tensor
        frames = list(self._frame_buffer)
        resized = [cv2.resize(f, (INPUT_W, INPUT_H)) for f in frames]
        stacked = np.concatenate([f.transpose(2, 0, 1) for f in resized], axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._tracknet(tensor)  # (1, 3, H, W) — 3 heatmaps for 3 frames
            heatmap = output[0, -1].cpu().numpy()  # use last frame's heatmap

        # Find peak in heatmap
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        if max_val < 0.5:  # below confidence threshold
            self.trail.append(None)
            return None

        # Scale back to original frame size
        orig_h, orig_w = self._frame_buffer[-1].shape[:2]
        cx = int(max_loc[0] * orig_w / INPUT_W)
        cy = int(max_loc[1] * orig_h / INPUT_H)
        r = max(4, int(orig_w * 0.012))  # estimated ball radius

        self.trail.append((cx, cy))
        return {
            "bbox": (cx - r, cy - r, cx + r, cy + r),
            "center": (cx, cy),
            "confidence": float(max_val),
        }

    # ------------------------------------------------------------------
    # ArtLabss TrackNet backend (ONNX Runtime + DirectML / Arc GPU)
    # ------------------------------------------------------------------

    def _load_artlabss(self, model_path: str) -> None:
        """
        Load TrackNet as an ONNX model via ONNX Runtime with DirectML (Arc GPU).
        Expects models/tracknet.onnx — run convert_to_onnx.py once to generate it.
        Falls back to CPU if DirectML is unavailable.
        """
        import onnxruntime as ort
        import os

        onnx_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "tracknet.onnx")
        onnx_path = os.path.normpath(onnx_path)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                "Run: python convert_to_onnx.py"
            )

        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if "DmlExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self._ort_session = ort.InferenceSession(onnx_path, providers=providers)
        self._ort_input_name = self._ort_session.get_inputs()[0].name

        self._artlabss_h = 360
        self._artlabss_w = 640
        self._artlabss_n_classes = 256

        device = "Arc GPU (DirectML)" if "DmlExecutionProvider" in providers else "CPU"
        print(f"[ArtLabss] Loaded TrackNet ONNX on {device}")

    def _detect_artlabss(self) -> dict | None:
        """
        Run ArtLabss TrackNet inference via ONNX Runtime.

        Input:  single frame → (1, 3, 360, 640) float32
        Output: (1, H*W, N_CLASSES) softmax → argmax → (H, W) heatmap
        Ball found via threshold + HoughCircles.
        """
        if not self._frame_buffer:
            self.trail.append(None)
            return None

        INPUT_W = self._artlabss_w
        INPUT_H = self._artlabss_h
        N_CLASSES = self._artlabss_n_classes

        frame = self._frame_buffer[-1]
        orig_h, orig_w = frame.shape[:2]

        img = cv2.resize(frame, (INPUT_W, INPUT_H)).astype(np.float32)
        X = np.rollaxis(img, 2, 0)          # (H,W,3) → (3,H,W)
        tensor = np.array([X])              # (1, 3, H, W)

        pr = self._ort_session.run(None, {self._ort_input_name: tensor})[0]
        # pr shape: (1, H*W, N_CLASSES)
        pr = pr[0].reshape((INPUT_H, INPUT_W, N_CLASSES)).argmax(axis=2).astype(np.uint8)

        heatmap = cv2.resize(pr, (orig_w, orig_h))
        _, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        circles = cv2.HoughCircles(
            heatmap, cv2.HOUGH_GRADIENT,
            dp=1, minDist=1, param1=50, param2=2,
            minRadius=2, maxRadius=7,
        )

        if circles is None or len(circles) != 1:
            self.trail.append(None)
            return None

        cx = int(circles[0][0][0])
        cy = int(circles[0][0][1])
        r  = max(int(circles[0][0][2]), 4)

        self.trail.append((cx, cy))
        return {
            "bbox": (cx - r, cy - r, cx + r, cy + r),
            "center": (cx, cy),
            "confidence": 1.0,
        }

