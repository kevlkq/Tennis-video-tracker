import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter


class BallSpeedCalculator:
    """
    Estimates ball speed using a Kalman filter for smoothing and
    an optional court homography for real-world distance conversion.
    Falls back to pixels/frame if no calibration is available.
    """

    def __init__(self, fps: float, history: int = 6):
        self.fps = fps
        self.history = history
        self._positions: deque = deque(maxlen=history)  # (frame_idx, x, y)
        self.current_speed_kph: float = 0.0
        self.current_speed_mph: float = 0.0
        self._frame_idx: int = 0
        self._kf = self._build_kalman()
        self._kf_initialized = False
        self._court_detector = None  # injected externally if available

    # ------------------------------------------------------------------

    def set_court_detector(self, court_detector) -> None:
        self._court_detector = court_detector

    def update(self, center: tuple[int, int] | None) -> float:
        """
        Feed one frame's ball centre (or None if not detected).
        Returns speed in km/h (0.0 if insufficient data).
        """
        self._frame_idx += 1

        if center is None:
            return self.current_speed_kph

        x, y = center

        # Kalman predict + update
        if not self._kf_initialized:
            self._kf.x = np.array([[x], [y], [0.], [0.]])
            self._kf_initialized = True
        else:
            self._kf.predict()
            self._kf.update(np.array([[x], [y]]))

        sx, sy = float(self._kf.x[0][0]), float(self._kf.x[1][0])
        self._positions.append((self._frame_idx, sx, sy))

        if len(self._positions) < 2:
            return self.current_speed_kph

        # Use oldest and newest smoothed position
        f0, x0, y0 = self._positions[0]
        f1, x1, y1 = self._positions[-1]
        frames_elapsed = f1 - f0
        if frames_elapsed == 0:
            return self.current_speed_kph

        px_dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Convert to real-world distance if calibrated
        if self._court_detector is not None:
            m_dist = self._court_detector.pixel_distance_to_metres((int(x0), int(y0)), (int(x1), int(y1)))
        else:
            m_dist = None

        time_sec = frames_elapsed / self.fps

        if m_dist is not None and m_dist > 0:
            speed_ms = m_dist / time_sec
        else:
            # Fallback: assume ~1 pixel ≈ 0.01 m (very rough, calibrate for accuracy)
            speed_ms = (px_dist * 0.01) / time_sec

        self.current_speed_kph = speed_ms * 3.6
        self.current_speed_mph = speed_ms * 2.237
        return self.current_speed_kph

    # ------------------------------------------------------------------

    def _build_kalman(self) -> KalmanFilter:
        """Constant-velocity 2-D Kalman filter: state = [x, y, vx, vy]."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        kf.R *= 10       # measurement noise
        kf.Q *= 0.1      # process noise
        kf.P *= 100
        return kf
