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
    # ArtLabss TrackNet backend (TensorFlow / Keras)
    # ------------------------------------------------------------------

    def _load_artlabss(self, model_path: str) -> None:
        """
        Build the ArtLabss TrackNet architecture then load weights-only file.

        The file ArtLabss/WeightsTracknet/model.1 is weights-only HDF5 —
        we must rebuild the architecture first, then call load_weights().
        Architecture is imported directly from the ArtLabss repo.
        """
        import sys
        import os
        artlabss_path = os.path.join(os.path.dirname(__file__), "..", "..", "ArtLabss")
        if artlabss_path not in sys.path:
            sys.path.insert(0, artlabss_path)

        try:
            from Models.tracknet import trackNet
        except ImportError:
            raise ImportError(
                "Cannot import ArtLabss trackNet. "
                "Make sure you cloned the repo: git clone https://github.com/ArtLabss/tennis-tracking.git ArtLabss"
            )

        # ArtLabss constants: 256 classes, 360x640 input
        self._artlabss_n_classes = 256
        self._artlabss_h = 360
        self._artlabss_w = 640

        self._tf_model = trackNet(
            self._artlabss_n_classes,
            input_height=self._artlabss_h,
            input_width=self._artlabss_w,
        )
        self._tf_model.compile(
            loss="categorical_crossentropy",
            optimizer="adadelta",
            metrics=["accuracy"],
        )
        # Keras 3 requires .h5 extension for legacy weight files.
        # If the file lacks it, make a renamed copy and load that instead.
        import shutil, tempfile, pathlib
        p = pathlib.Path(model_path)
        if p.suffix.lower() != ".h5":
            tmp = pathlib.Path(tempfile.mkdtemp()) / (p.stem + ".h5")
            shutil.copy2(p, tmp)
            load_path = str(tmp)
            print(f"[ArtLabss] Renamed weights to temp .h5 for Keras 3 compatibility")
        else:
            load_path = model_path

        self._tf_model.load_weights(load_path)
        print(f"[ArtLabss] Loaded TrackNet weights from {model_path}")

    def _detect_artlabss(self) -> dict | None:
        """
        Run ArtLabss TrackNet inference on the current frame.

        Input:  single frame, channels_first → (1, 3, 360, 640)
        Output: (H*W, n_classes) softmax → argmax → (H, W) grayscale heatmap
        Ball found via threshold + HoughCircles (matches original ArtLabss logic).
        """
        if not self._frame_buffer:
            self.trail.append(None)
            return None

        INPUT_W = self._artlabss_w   # 640
        INPUT_H = self._artlabss_h   # 360
        N_CLASSES = self._artlabss_n_classes  # 256

        frame = self._frame_buffer[-1]
        orig_h, orig_w = frame.shape[:2]

        img = cv2.resize(frame, (INPUT_W, INPUT_H)).astype(np.float32)
        # channels_first: (H, W, 3) → (3, H, W)
        X = np.rollaxis(img, 2, 0)
        tensor = np.array([X])  # (1, 3, H, W)

        pr = self._tf_model.predict(tensor, verbose=0)[0]
        # pr shape: (H*W, N_CLASSES) → (H, W, N_CLASSES) → argmax → (H, W)
        pr = pr.reshape((INPUT_H, INPUT_W, N_CLASSES)).argmax(axis=2).astype(np.uint8)

        # Scale heatmap back to original frame size
        heatmap = cv2.resize(pr, (orig_w, orig_h))
        _, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        # Find ball circle in heatmap
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
        r = max(int(circles[0][0][2]), 4)

        self.trail.append((cx, cy))
        return {
            "bbox": (cx - r, cy - r, cx + r, cy + r),
            "center": (cx, cy),
            "confidence": float(max_val),
        }
