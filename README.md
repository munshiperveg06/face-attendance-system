# AI-Powered Face Recognition Attendance System

Contactless attendance: a camera sees an employee, the system matches the face
against enrolled employees and records a timestamped check-in in SQLite.
Implements the pipeline from the project slides: face detection -> encoding ->
distance-based recognition -> database -> HR dashboard.

## Quick start (Windows / macOS / Linux, Python 3.9+)

```bash
python -m venv venv
venv\Scripts\activate            # Windows      (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open **http://127.0.0.1:5000** (must be `127.0.0.1`/`localhost` or https, or the browser blocks the camera).

1. **Employees** -> add ID + name -> the enroll page opens -> *Start auto-capture* (15 photos,
   turn your head slightly) -- or upload photos.
2. **Live Kiosk** -> stand in front of the camera for ~2 s -> attendance is recorded.
3. **Dashboard** shows present / absent / late; **Reports** filters by date and exports **CSV**.

### Desktop (OpenCV window) alternative
```bash
python enroll.py --id E001 --name "Ayan Roy"      # SPACE = capture, a = auto, q = done
python live_attendance.py                          # or --camera rtsp://user:pass@ip/stream for CCTV
```

## How it works
| Slide topic | Implementation |
|---|---|
| Face detection | Haar cascade (OpenCV) - or dlib HOG when installed |
| Face encoding | LBPH histograms (default) - or dlib 128-d vectors |
| Recognition | LBPH distance / Euclidean distance vs. a threshold (`config.py`) |
| Database | SQLite (`attendance.db`), one row per employee per day, CSV export |
| Dashboard | Flask web UI |

**Anti-false-positive logic:** a person must be matched on `CONFIRM_FRAMES` (3) consecutive
frames before attendance is written; one check-in per day; later sightings (after
`MIN_CHECKOUT_MINUTES`) update "last seen"; 60 s cooldown per person.

## Optional: higher accuracy (dlib backend)
```bash
pip install cmake dlib face_recognition setuptools
```
Detected automatically (`FACE_BACKEND=auto`). Windows needs "Desktop development with C++"
from Visual Studio Build Tools first. After switching backend, no re-enrollment is needed -
training re-reads the stored photos.

## Tuning (`config.py`)
* Unknown people being accepted -> lower `LBPH_THRESHOLD` (e.g. 60) / `DLIB_TOLERANCE` (e.g. 0.45).
* Real employees rejected -> raise it (e.g. 80) or enroll more photos in varied lighting.
* Work start time for "Late": `WORK_START` (default 09:30).

## Tests
```bash
pip install scikit-image        # only for the image-based tests
python -m unittest discover -s tests -v
```

## Security & privacy notes
* Face photos live in `dataset/` - they are biometric data. Get employee consent, restrict access to
  this folder, and follow your local law (e.g. India's DPDP Act, GDPR).
* Face recognition can be fooled by a printed photo or screen. This project has **no liveness
  detection**; do not use it alone for high-security decisions.
* The dashboard has no login. Keep it on localhost/internal network, or put it behind authentication
  and set `SECRET_KEY` before exposing it.

## Project layout
```
app.py              Flask dashboard + browser kiosk
recognizer.py       detection/recognition engine (LBPH + optional dlib) and frame tracker
database.py         SQLite schema and attendance logic
config.py           all tunable settings
enroll.py / live_attendance.py   desktop OpenCV tools
templates/ static/  web UI
tests/              end-to-end tests
```
