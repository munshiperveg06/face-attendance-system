"""Face detection + recognition engine with two interchangeable backends.

lbph : OpenCV Haar-cascade detection + LBPH recognition. Pure OpenCV, installs
       anywhere with `pip install opencv-contrib-python`.
dlib : face_recognition library (HOG detection + 128-d encodings compared with
       Euclidean distance). More accurate, but needs CMake/C++ to install.
"""
import pickle
import shutil
import threading
import time

import cv2
import numpy as np

import config

try:  # optional dependency
    import face_recognition
    HAVE_DLIB = True
except Exception:
    face_recognition = None
    HAVE_DLIB = False

FACE_SIZE = 200
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def resolve_backend():
    choice = config.BACKEND.lower()
    if choice == "auto":
        choice = "dlib" if HAVE_DLIB else "lbph"
    if choice == "dlib" and not HAVE_DLIB:
        raise RuntimeError("FACE_BACKEND=dlib but face_recognition is not installed.")
    if choice == "lbph" and not hasattr(cv2, "face"):
        raise RuntimeError("LBPH needs the contrib build: pip uninstall opencv-python && "
                           "pip install opencv-contrib-python")
    if choice not in ("lbph", "dlib"):
        raise RuntimeError(f"Unknown FACE_BACKEND '{choice}'")
    return choice


def _resize_max(img, max_w=640):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)


def detect_faces(frame_bgr):
    """Haar detection. Returns [(x, y, w, h)] ranked by detector confidence.

    Haar cascades occasionally fire on background clutter, and such a false hit
    can be *larger* than the real face, so we rank by confidence (not area) and
    drop candidates far weaker than the best one."""
    gray = _CLAHE.apply(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    m = config.MIN_FACE_SIZE
    boxes, _, weights = _CASCADE.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(m, m), outputRejectLevels=True)
    cands = sorted(((float(w), tuple(int(v) for v in b)) for b, w in zip(boxes, np.ravel(weights))),
                   reverse=True)
    if not cands:
        return []
    best = cands[0][0]
    return [b for w, b in cands if w >= 2.5 and w >= 0.55 * best] or [cands[0][1]]


def _prep_lbph(gray_crop):
    face = cv2.resize(gray_crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    return _CLAHE.apply(face)


def _augment(face):
    """Cheap augmentation so a handful of photos generalises better."""
    out = [face, cv2.flip(face, 1)]
    for alpha, beta in ((1.15, 12), (0.85, -12)):
        out.append(cv2.convertScaleAbs(face, alpha=alpha, beta=beta))
    return out


class FaceEngine:
    def __init__(self):
        self.backend = resolve_backend()
        self.lock = threading.RLock()
        self.model = None            # LBPH recognizer
        self.labels = []             # LBPH int label -> emp_id
        self.known_enc = np.empty((0, 128))   # dlib
        self.known_ids = []
        self.train()

    # ------------------------------------------------------------ dataset
    def employee_dir(self, emp_id):
        return config.DATASET_DIR / emp_id

    def image_paths(self, emp_id):
        d = self.employee_dir(emp_id)
        return sorted(d.glob("*.jpg")) if d.exists() else []

    def sample_count(self, emp_id):
        return len(self.image_paths(emp_id))

    def add_sample(self, emp_id, frame_bgr):
        """Validate + store one enrollment photo. Returns (ok, message)."""
        if frame_bgr is None:
            return False, "Could not read image."
        frame_bgr = _resize_max(frame_bgr)
        with self.lock:
            faces = self._locate(frame_bgr)
        if not faces:
            return False, "No face detected - face the camera in good light."
        if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.4 * faces[0][2] * faces[0][3]:
            return False, "More than one face in view - only one person please."
        d = self.employee_dir(emp_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True, "Sample saved."

    def remove_employee(self, emp_id):
        shutil.rmtree(self.employee_dir(emp_id), ignore_errors=True)

    # ----------------------------------------------------------- detection
    def _locate(self, frame_bgr):
        """Face boxes as (x, y, w, h) using the active backend's detector."""
        if self.backend == "dlib":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            boxes = [(l, t, r - l, b - t) for (t, r, b, l) in locs]
            boxes = [b for b in boxes if min(b[2], b[3]) >= config.MIN_FACE_SIZE]
            return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        return detect_faces(frame_bgr)

    # ------------------------------------------------------------ training
    def train(self):
        with self.lock:
            from database import list_employees
            ids = [e["emp_id"] for e in list_employees()]
            if self.backend == "dlib":
                self._train_dlib(ids)
            else:
                self._train_lbph(ids)

    def _train_lbph(self, ids):
        faces, labels, self.labels = [], [], []
        for emp_id in ids:
            for p in self.image_paths(emp_id):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                boxes = detect_faces(img)
                if not boxes:
                    continue
                x, y, w, h = boxes[0]
                face = _prep_lbph(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w])
                if emp_id not in self.labels:
                    self.labels.append(emp_id)
                for f in _augment(face):
                    faces.append(f)
                    labels.append(self.labels.index(emp_id))
        if not faces:
            self.model = None
            return
        self.model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        self.model.train(faces, np.array(labels))

    def _train_dlib(self, ids):
        cache_file = config.MODEL_DIR / "dlib_cache.pkl"
        try:
            cache = pickle.loads(cache_file.read_bytes())
        except Exception:
            cache = {}
        encs, owners, fresh = [], [], {}
        for emp_id in ids:
            for p in self.image_paths(emp_id):
                key = (str(p), p.stat().st_mtime_ns)
                enc = cache.get(key)
                if enc is None:
                    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                    e = face_recognition.face_encodings(img, num_jitters=1)
                    enc = e[0] if e else False
                fresh[key] = enc
                if enc is not False:
                    encs.append(enc)
                    owners.append(emp_id)
        cache_file.write_bytes(pickle.dumps(fresh))
        self.known_enc = np.array(encs) if encs else np.empty((0, 128))
        self.known_ids = owners

    # --------------------------------------------------------- recognition
    def identify(self, frame_bgr):
        """Returns [{'box': (x,y,w,h), 'emp_id': str|None, 'confidence': 0-100}, ...]."""
        frame_bgr = _resize_max(frame_bgr)
        with self.lock:
            boxes = self._locate(frame_bgr)
            if self.backend == "dlib":
                return self._identify_dlib(frame_bgr, boxes)
            return self._identify_lbph(frame_bgr, boxes)

    def _identify_lbph(self, frame, boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []
        for (x, y, w, h) in boxes:
            emp_id, conf = None, 0.0
            if self.model is not None:
                label, dist = self.model.predict(_prep_lbph(gray[y:y + h, x:x + w]))
                conf = float(np.clip(100 - dist, 0, 100))
                if dist <= config.LBPH_THRESHOLD:
                    emp_id = self.labels[label]
            results.append({"box": (x, y, w, h), "emp_id": emp_id, "confidence": round(conf, 1)})
        return results

    def _identify_dlib(self, frame, boxes):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = [(y, x + w, y + h, x) for (x, y, w, h) in boxes]
        encs = face_recognition.face_encodings(rgb, locs) if locs else []
        results = []
        for box, enc in zip(boxes, encs):
            emp_id, conf = None, 0.0
            if len(self.known_enc):
                dists = np.linalg.norm(self.known_enc - enc, axis=1)
                i = int(np.argmin(dists))
                conf = float(np.clip((1 - dists[i]) * 100, 0, 100))
                if dists[i] <= config.DLIB_TOLERANCE:
                    emp_id = self.known_ids[i]
            results.append({"box": box, "emp_id": emp_id, "confidence": round(conf, 1)})
        return results


class Tracker:
    """Requires N consecutive sightings and applies a per-person cooldown, so a
    single noisy frame can never mark someone present."""

    def __init__(self):
        self.streak = {}      # emp_id -> (count, last_seen)
        self.cooldown = {}    # emp_id -> unix time until which we ignore them

    def update(self, results, now=None):
        now = now or time.time()
        confirmed = []
        for emp_id in {r["emp_id"] for r in results if r["emp_id"]}:
            if now < self.cooldown.get(emp_id, 0):
                continue
            count, last = self.streak.get(emp_id, (0, 0))
            count = count + 1 if now - last <= config.STREAK_TIMEOUT else 1
            self.streak[emp_id] = (count, now)
            if count >= config.CONFIRM_FRAMES:
                confirmed.append(emp_id)
                self.streak.pop(emp_id, None)
                self.cooldown[emp_id] = now + config.COOLDOWN_SECONDS
        return confirmed
