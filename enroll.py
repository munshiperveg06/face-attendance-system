"""Enroll an employee from a desktop webcam.
   python enroll.py --id E001 --name "Ayan Roy" [--dept CSE] [--camera 0]
Press SPACE to capture a photo (or 'a' to toggle auto-capture), 'q' to finish."""
import argparse
import sys
import time

import cv2

import config
import database as db
from recognizer import FaceEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dept", default="")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    db.init_db()
    if db.get_employee(a.id) is None:
        try:
            db.add_employee(a.id, a.name, a.dept)
        except ValueError as e:
            sys.exit(str(e))
    engine = FaceEngine()

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {a.camera}. Try --camera 1, or enroll through the web UI.")
    auto, last, msg = False, 0, "SPACE = capture, a = auto, q = done"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shown = frame.copy()
        for r in engine.identify(frame):
            x, y, w, h = r["box"]
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
        n = engine.sample_count(a.id)
        cv2.putText(shown, f"{a.name}: {n}/{config.SAMPLES_REQUIRED}  {msg}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Enroll", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto = not auto
        if key == ord(" ") or (auto and time.time() - last > 0.8):
            ok2, msg = engine.add_sample(a.id, frame)
            last = time.time()
            if auto and engine.sample_count(a.id) >= config.SAMPLES_REQUIRED:
                break
    cap.release()
    cv2.destroyAllWindows()
    engine.train()
    print(f"Saved {engine.sample_count(a.id)} sample(s) for {a.name}.")


if __name__ == "__main__":
    main()
