"""
Tennis Video Tracker — main pipeline entry point.

Usage:
    python src/main.py --input path/to/video.mp4
    python src/main.py --input 0          # webcam
    python src/main.py --input video.mp4 --output out.mp4
"""

import argparse
import sys
import cv2
import numpy as np

from detection import BallDetector, PlayerDetector, CourtDetector
from tracking import BallSpeedCalculator
from overlay import Renderer


def parse_args():
    p = argparse.ArgumentParser(description="Tennis Video Tracker")
    p.add_argument("--input", required=True, help="Video file path or camera index (0)")
    p.add_argument("--output", default=None, help="Optional output video file path")
    p.add_argument("--ball-model", default="yolov8n.pt", help="Model weights for ball detection")
    p.add_argument("--backend", default="yolo", choices=["yolo", "tracknetv3", "artlabss"],
                   help="Ball detection backend")
    p.add_argument("--no-skeleton", action="store_true", help="Disable pose skeleton overlay")
    p.add_argument("--no-court", action="store_true", help="Disable court line overlay")
    p.add_argument("--no-speed", action="store_true", help="Disable speed HUD")
    p.add_argument("--calibrate", action="store_true",
                   help="Interactive court calibration: click 4 corners on the first frame "
                        "to enable real-world speed (essential at low camera angles)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Downscale frames before detection (e.g. 0.5 = half resolution, faster on CPU)")
    return p.parse_args()


def run_court_calibration(frame: np.ndarray, court_det) -> None:
    """
    Opens an interactive window where the user clicks the 4 court corners
    to calibrate the homography for real-world speed calculation.

    Click order:
      1. Top-left baseline corner
      2. Top-right baseline corner
      3. Bottom-right baseline corner
      4. Bottom-left baseline corner

    Press ENTER or 'c' to confirm, ESC to skip calibration.
    """
    WINDOW = "Court Calibration"
    corners: list[tuple[int, int]] = []

    # Scale frame to fit screen (max 1280x720 for the calibration window)
    MAX_W, MAX_H = 1280, 720
    h_orig, w_orig = frame.shape[:2]
    scale = min(MAX_W / w_orig, MAX_H / h_orig, 1.0)
    disp_w = int(w_orig * scale)
    disp_h = int(h_orig * scale)
    display = cv2.resize(frame, (disp_w, disp_h)) if scale < 1.0 else frame.copy()

    CORNER_LABELS = [
        "1: Top-Left",
        "2: Top-Right",
        "3: Bottom-Right",
        "4: Bottom-Left",
    ]
    CORNER_COLOURS = [
        (0, 255, 255),   # yellow
        (0, 165, 255),   # orange
        (0, 100, 255),   # red-orange
        (255, 100, 0),   # blue
    ]

    def mouse_cb(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            # Map display coords back to original frame coords
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            corners.append((orig_x, orig_y))
            idx = len(corners) - 1
            # Draw on display using display-space coords
            cv2.circle(display, (x, y), 8, CORNER_COLOURS[idx], -1, cv2.LINE_AA)
            cv2.circle(display, (x, y), 8, (255, 255, 255), 2, cv2.LINE_AA)
            label = CORNER_LABELS[idx]
            cv2.putText(display, label, (x + 10, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CORNER_COLOURS[idx], 2, cv2.LINE_AA)
            if idx > 0:
                prev_disp = (int(corners[idx - 1][0] * scale), int(corners[idx - 1][1] * scale))
                cv2.line(display, prev_disp, (x, y), (200, 200, 200), 1, cv2.LINE_AA)
            if len(corners) == 4:
                first_disp = (int(corners[0][0] * scale), int(corners[0][1] * scale))
                cv2.line(display, (x, y), first_disp, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(display, "Press ENTER to confirm, ESC to skip",
                            (20, display.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    instruction = "Click 4 court corners: TL -> TR -> BR -> BL   |   ESC = skip"
    next_label_y = 30

    print("[CALIBRATION] Click the 4 court corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    print("[CALIBRATION] Press ENTER/c to confirm, ESC to skip.")

    cv2.resizeWindow(WINDOW, disp_w, disp_h)

    while True:
        view = display.copy()
        cv2.putText(view, instruction, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if len(corners) < 4:
            next_label = f"Next click -> {CORNER_LABELS[len(corners)]}"
            cv2.putText(view, next_label, (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CORNER_COLOURS[len(corners)], 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC — skip
            print("[CALIBRATION] Skipped. Speed will use pixel fallback (~10-15% error).")
            break
        if key in (13, ord("c")) and len(corners) == 4:  # ENTER or 'c'
            ok = court_det.calibrate(frame, corners)
            if ok:
                print("[CALIBRATION] Homography set — speed will use real-world metres.")
            else:
                print("[CALIBRATION] Calibration failed (bad corners?). Using pixel fallback.")
            break

    cv2.destroyWindow(WINDOW)


def open_capture(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)
    return cap


def main():
    args = parse_args()

    cap = open_capture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {args.input}  |  {w}x{h} @ {fps:.1f} fps")

    # --- Components ---
    ball_det = BallDetector(backend=args.backend, model_path=args.ball_model, confidence=0.3)
    player_det = PlayerDetector(model_path="yolov8n-pose.pt", confidence=0.4)
    court_det = CourtDetector()
    speed_calc = BallSpeedCalculator(fps=fps)
    speed_calc.set_court_detector(court_det)  # enables real-world speed if calibrated

    renderer = Renderer(
        show_skeleton=not args.no_skeleton,
        show_court=not args.no_court,
        show_speed=not args.no_speed,
    )

    # --- Optional output writer ---
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    # --- Interactive calibration (highly recommended at low camera angles) ---
    if args.calibrate:
        ret, first_frame = cap.read()
        if ret:
            run_court_calibration(first_frame, court_det)
            # Rewind to start so we don't skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("[WARN] Could not read first frame for calibration.")

    import time

    offline = bool(args.output)  # no display window when saving to file

    if not offline:
        DISPLAY_W, DISPLAY_H = 1280, 720
        cv2.namedWindow("Tennis Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tennis Tracker", DISPLAY_W, DISPLAY_H)

    print("[INFO] Press 'q' to quit." if not offline else
          f"[INFO] Processing offline → {args.output}  (Ctrl+C to cancel)")

    paused = False
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        proc = cv2.resize(frame, None, fx=args.scale, fy=args.scale) if args.scale != 1.0 else frame

        ball         = ball_det.detect(proc)
        players      = player_det.detect(proc)
        court_lines  = court_det.detect_lines(proc)
        center       = ball["center"] if ball else None
        speed_kph    = speed_calc.update(center)

        annotated = renderer.render(
            frame,
            ball=ball,
            ball_trail=ball_det.get_trail(),
            players=players,
            court_lines=court_lines,
            speed_kph=speed_kph,
            speed_mph=speed_calc.current_speed_mph,
        )

        if writer:
            writer.write(annotated)

        if offline:
            elapsed = time.time() - t_start
            pct = frame_count / total_frames * 100 if total_frames else 0
            fps_proc = frame_count / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_count) / fps_proc if fps_proc > 0 else 0
            print(f"\r[{pct:5.1f}%] frame {frame_count}/{total_frames} "
                  f"| {fps_proc:.2f} fps | ETA {eta:.0f}s   ", end="", flush=True)
        else:
            cv2.imshow("Tennis Tracker", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                fname = f"screenshot_{frame_count}.png"
                cv2.imwrite(fname, annotated)
                print(f"\n[INFO] Saved {fname}")

    if offline:
        print()  # newline after progress
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
