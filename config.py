"""Central settings. Edit values here or override with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"          # dataset/<employee_id>/*.jpg
MODEL_DIR = BASE_DIR / "models"             # caches (dlib encodings)
DB_PATH = BASE_DIR / "attendance.db"

# "auto" -> dlib if face_recognition is installed, otherwise OpenCV LBPH.
# Force one with FACE_BACKEND=lbph or FACE_BACKEND=dlib
BACKEND = os.environ.get("FACE_BACKEND", "auto")

# LBPH: distance, LOWER is better. 55-80 is a sensible range.
LBPH_THRESHOLD = float(os.environ.get("LBPH_THRESHOLD", 70))
# dlib: Euclidean distance between 128-d encodings. 0.6 is the library default,
# 0.5 is stricter (fewer false accepts).
DLIB_TOLERANCE = float(os.environ.get("DLIB_TOLERANCE", 0.5))

MIN_FACE_SIZE = 70          # px - ignore faces smaller than this
SAMPLES_REQUIRED = 15       # photos per employee for good accuracy
CONFIRM_FRAMES = 3          # consecutive matches needed before marking attendance
STREAK_TIMEOUT = 3.0        # seconds - streak resets if face not seen for this long
COOLDOWN_SECONDS = 60       # don't re-process the same person within this window
MIN_CHECKOUT_MINUTES = 30   # a later sighting counts as check-out only after this
WORK_START = os.environ.get("WORK_START", "09:30")   # check-in after this = Late

DATASET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
