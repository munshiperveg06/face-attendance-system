"""Desktop live attendance using an OpenCV window (webcam or RTSP/CCTV URL).
   python live_attendance.py [--camera 0 | --camera rtsp://user:pass@ip/stream]
Press 'q' to quit."""
import argparse
import sys
import time

import cv2

import database as db
from recognizer import FaceEngine, Tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0", help="index (0,1..) or stream URL")
    a = ap.parse_args()
    src = int(a.camera) if a.camera.isdigit() else a.camera

    db.init_db()
    engine, tracker = FaceEngine(), Tracker()
    names = {e["emp_id"]: e["name"] for e in db.list_employees()}
    if not names:
        sys.exit("No employees registered. Run enroll.py or use the web dashboard first.")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera '{a.camera}'.")
    banner, banner_until, results, n = "", 0, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % 3 == 0:                      # process every 3rd frame for speed
            results = engine.identify(frame)
            for emp_id in tracker.update(results):
                res = db.mark_attendance(emp_id)
                banner = f"{res.get('name', emp_id)}: {res['status'].replace('_', ' ')} {res.get('time', '')}"
                banner_until = time.time() + 4
                print(time.strftime("%H:%M:%S"), banner)
        for r in results:
            x, y, w, h = r["box"]
            colour = (0, 220, 0) if r["emp_id"] else (0, 0, 255)
            label = f"{names.get(r['emp_id'], 'Unknown')} {r['confidence']:.0f}%" if r["emp_id"] else "Unknown"
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        if time.time() < banner_until:
            cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Attendance (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
