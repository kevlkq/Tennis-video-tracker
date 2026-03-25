import cv2
import numpy as np


# Real tennis court dimensions (metres) used for calibration
COURT_LENGTH_M = 23.77
COURT_WIDTH_M = 10.97   # doubles width
COURT_WIDTH_SINGLES_M = 8.23


class CourtDetector:
    """
    Detects court lines via Hough transform and estimates a homography
    so we can convert pixel distances to real-world metres.
    """

    def __init__(self):
        self.homography: np.ndarray | None = None  # pixel -> metres transform
        self._cached_lines: list | None = None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_lines(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x1,y1,x2,y2) line segments for court markings."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Only look in the bottom 65% of the frame — court is never in the sky
        roi_y = int(h * 0.35)
        roi = gray[roi_y:, :]

        # Enhance white court lines only (very bright pixels)
        _, thresh = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        lines_raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=120,        # higher = fewer false positives
            minLineLength=80,     # ignore short stray lines
            maxLineGap=15,
        )

        if lines_raw is None:
            self._cached_lines = []
            return []

        lines = []
        for l in lines_raw:
            x1, y1, x2, y2 = l[0]
            # Filter by angle: keep only near-horizontal (≤30°) or near-vertical (≥60°)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle > 30 and angle < 60:
                continue
            # Shift y coords back to full frame space
            lines.append((x1, y1 + roi_y, x2, y2 + roi_y))

        self._cached_lines = lines
        return lines

    # ------------------------------------------------------------------
    # Calibration — call once on a clear frame where all 4 baseline
    # corners are visible so we can build a homography.
    # ------------------------------------------------------------------

    def calibrate(self, frame: np.ndarray, court_corners_px: list[tuple[int, int]]) -> bool:
        """
        court_corners_px: 4 pixel points in order:
          [top-left, top-right, bottom-right, bottom-left] of the baseline rectangle.
        Builds a homography matrix for pixel -> metres conversion.
        Returns True on success.
        """
        if len(court_corners_px) != 4:
            return False

        dst_pts = np.array([
            [0,               0],
            [COURT_LENGTH_M,  0],
            [COURT_LENGTH_M,  COURT_WIDTH_M],
            [0,               COURT_WIDTH_M],
        ], dtype=np.float32)

        src_pts = np.array(court_corners_px, dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        if H is not None:
            self.homography = H
            return True
        return False

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def pixels_to_metres(self, px: tuple[int, int]) -> tuple[float, float] | None:
        """Convert a pixel (x, y) to court metres using the calibrated homography."""
        if self.homography is None:
            return None
        pt = np.array([[px]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.homography)
        return float(result[0][0][0]), float(result[0][0][1])

    def pixel_distance_to_metres(self, p1: tuple[int, int], p2: tuple[int, int]) -> float | None:
        """Euclidean distance between two pixel points converted to metres."""
        m1 = self.pixels_to_metres(p1)
        m2 = self.pixels_to_metres(p2)
        if m1 is None or m2 is None:
            return None
        return float(np.sqrt((m2[0] - m1[0]) ** 2 + (m2[1] - m1[1]) ** 2))
