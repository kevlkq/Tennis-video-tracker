import cv2
import numpy as np
from detection.player_detector import COCO_SKELETON, BODY_INDICES


# Colour palette (BGR)
COLOUR_BALL = (0, 255, 255)        # yellow
COLOUR_BALL_TRAIL = (0, 200, 200)
COLOUR_COURT = (0, 255, 0)         # green
COLOUR_SKELETON = (255, 100, 0)    # blue-orange
COLOUR_BBOX = (255, 60, 60)        # player box
COLOUR_HUD_BG = (0, 0, 0)
COLOUR_HUD_TEXT = (255, 255, 255)
COLOUR_SPEED = (0, 200, 255)       # orange-yellow for speed

POSE_CONNECTIONS = COCO_SKELETON


class Renderer:
    """Draws all detections onto a frame and returns the annotated copy."""

    def __init__(self, show_bbox: bool = True, show_skeleton: bool = True,
                 show_court: bool = True, show_ball: bool = True,
                 show_speed: bool = True):
        self.show_bbox = show_bbox
        self.show_skeleton = show_skeleton
        self.show_court = show_court
        self.show_ball = show_ball
        self.show_speed = show_speed

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        ball: dict | None,
        ball_trail: list,
        players: list[dict],
        court_lines: list[tuple],
        speed_kph: float,
        speed_mph: float,
    ) -> np.ndarray:
        out = frame.copy()

        if self.show_court:
            self._draw_court_lines(out, court_lines)

        if self.show_ball:
            self._draw_ball(out, ball, ball_trail)

        for player in players:
            if self.show_bbox:
                self._draw_player_bbox(out, player["bbox"])
            if self.show_skeleton and player["landmarks"]:
                self._draw_skeleton(out, player["landmarks"])

        if self.show_speed:
            self._draw_speed_hud(out, speed_kph, speed_mph)

        return out

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_court_lines(self, frame: np.ndarray, lines: list) -> None:
        for x1, y1, x2, y2 in lines:
            cv2.line(frame, (x1, y1), (x2, y2), COLOUR_COURT, 2, cv2.LINE_AA)

    def _draw_ball(self, frame: np.ndarray, ball: dict | None, trail: list) -> None:
        # Draw fading trail
        for i, pt in enumerate(trail):
            if pt is None:
                continue
            alpha = int(255 * (i + 1) / len(trail))
            radius = max(2, int(6 * (i + 1) / len(trail)))
            color = (0, alpha, alpha)
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

        # Draw current ball
        if ball:
            cx, cy = ball["center"]
            cv2.circle(frame, (cx, cy), 8, COLOUR_BALL, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, COLOUR_BALL, -1, cv2.LINE_AA)

    def _draw_player_bbox(self, frame: np.ndarray, bbox: tuple) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOUR_BBOX, 2, cv2.LINE_AA)

    def _draw_skeleton(self, frame: np.ndarray, landmarks: list) -> None:
        # --- Glowing silhouette outline ---
        self._draw_player_silhouette(frame, landmarks)

        # --- Skeleton connections (skip if either endpoint is None / low confidence) ---
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]
            if p1 is None or p2 is None:
                continue
            cv2.line(frame, p1, p2, COLOUR_SKELETON, 2, cv2.LINE_AA)

        # --- Joints ---
        for pt in landmarks:
            if pt is None:
                continue
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, COLOUR_SKELETON, 1, cv2.LINE_AA)

    def _draw_player_silhouette(self, frame: np.ndarray, landmarks: list) -> None:
        """
        Draws a soft glowing outline around the player body using the
        convex hull of their pose landmarks — SwingVision-style.
        Uses COCO body keypoints (shoulders → ankles), skips face.
        """
        pts = np.array(
            [landmarks[i] for i in BODY_INDICES
             if i < len(landmarks) and landmarks[i] is not None],
            dtype=np.int32,
        )
        if len(pts) < 3:
            return

        hull = cv2.convexHull(pts)

        # Draw 3 progressively thinner layers for a glow effect
        glow_layers = [
            (12, (80,  40,  10), 0.25),   # wide, dark outer glow
            (6,  (160, 80,  20), 0.40),   # mid glow
            (2,  (255, 130, 50), 0.90),   # sharp edge
        ]
        overlay = frame.copy()
        for thickness, colour, alpha in glow_layers:
            cv2.polylines(overlay, [hull], isClosed=True, color=colour,
                          thickness=thickness, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            overlay = frame.copy()

    def _draw_speed_hud(self, frame: np.ndarray, kph: float, mph: float) -> None:
        h, w = frame.shape[:2]
        # Semi-transparent HUD box in top-right
        box_w, box_h = 200, 70
        x1, y1 = w - box_w - 10, 10
        x2, y2 = w - 10, y1 + box_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "BALL SPEED", (x1 + 10, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_HUD_TEXT, 1, cv2.LINE_AA)

        speed_str = f"{kph:.1f} km/h" if kph > 0 else "-- km/h"
        cv2.putText(frame, speed_str, (x1 + 10, y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOUR_SPEED, 2, cv2.LINE_AA)

        mph_str = f"({mph:.1f} mph)"
        cv2.putText(frame, mph_str, (x1 + 10, y1 + 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_HUD_TEXT, 1, cv2.LINE_AA)
