import cv2
import numpy as np
from ultralytics import YOLO


# COCO 17-keypoint skeleton connections (used by YOLOv8-pose)
COCO_SKELETON = [
    (5, 6),   # left shoulder - right shoulder
    (5, 7),   # left shoulder - left elbow
    (7, 9),   # left elbow - left wrist
    (6, 8),   # right shoulder - right elbow
    (8, 10),  # right elbow - right wrist
    (5, 11),  # left shoulder - left hip
    (6, 12),  # right shoulder - right hip
    (11, 12), # left hip - right hip
    (11, 13), # left hip - left knee
    (13, 15), # left knee - left ankle
    (12, 14), # right hip - right knee
    (14, 16), # right knee - right ankle
]

# Body keypoint indices used for silhouette hull (skip face)
BODY_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


class PlayerDetector:
    """
    Detects players and their pose keypoints using YOLOv8-pose.
    One model does both bounding box + skeleton — no mediapipe needed.

    Model auto-downloads on first run:
      yolov8n-pose.pt  (~6 MB,  fastest)
      yolov8s-pose.pt  (~23 MB, better accuracy)
    """

    POSE_CONNECTIONS = COCO_SKELETON

    def __init__(self, model_path: str = "yolov8n-pose.pt", confidence: float = 0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of dicts per detected player:
          - bbox:      (x1, y1, x2, y2)
          - landmarks: list of (x, y) pixel coords for 17 COCO keypoints, or None
        """
        results = self.model(frame, verbose=False, conf=self.confidence)
        players = []

        if not results or results[0].keypoints is None:
            return players

        boxes = results[0].boxes
        keypoints = results[0].keypoints.xy.cpu().numpy()   # (N, 17, 2)
        scores = results[0].keypoints.conf                  # (N, 17) confidence per keypoint

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            kp = keypoints[i]  # (17, 2)
            kp_conf = scores[i].cpu().numpy() if scores is not None else None

            # Only include keypoints with sufficient confidence
            landmarks = []
            for j, (kx, ky) in enumerate(kp):
                if kp_conf is not None and kp_conf[j] < 0.3:
                    landmarks.append(None)   # low confidence → skip this joint
                else:
                    landmarks.append((int(kx), int(ky)))

            players.append({"bbox": (x1, y1, x2, y2), "landmarks": landmarks})

        return players
