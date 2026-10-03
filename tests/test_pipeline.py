"""End-to-end tests. Run:  python -m unittest discover -s tests -v
Uses scikit-image's bundled astronaut photo when available; otherwise the
image-based tests are skipped (the database/tracker tests always run)."""
import base64
import importlib
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

TMP = Path(tempfile.mkdtemp())
config.DATASET_DIR, config.MODEL_DIR, config.DB_PATH = TMP / "ds", TMP / "models", TMP / "t.db"
config.DATASET_DIR.mkdir(), config.MODEL_DIR.mkdir()
config.BACKEND = os.environ.get("FACE_BACKEND", "lbph")

import database as db  # noqa: E402

try:
    from skimage import data as skdata
    FACE = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
except Exception:
    FACE = None


def variant(img, seed):
    """Simulate a different capture: lighting, small shift/rotation/scale, noise."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-5, 5), rng.uniform(0.92, 1.08))
    M[:, 2] += rng.uniform(-8, 8, 2)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-15, 15))
    noise = rng.normal(0, 4, out.shape)
    return np.clip(out.astype(float) + noise, 0, 255).astype(np.uint8)


def data_url(img):
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()


class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_employee_validation_and_duplicates(self):
        db.add_employee("D1", "Dee", "QA")
        with self.assertRaises(ValueError):
            db.add_employee("D1", "Again")
        with self.assertRaises(ValueError):
            db.add_employee("../evil", "Bad")
        with self.assertRaises(ValueError):
            db.add_employee("D2", "  ")

    def test_checkin_once_then_checkout_and_late_flag(self):
        db.add_employee("D3", "Dev")
        t0 = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        self.assertEqual(db.mark_attendance("D3", 90, now=t0)["status"], "check_in")
        self.assertEqual(db.mark_attendance("D3", 90, now=t0 + timedelta(minutes=5))["status"], "already")
        self.assertEqual(db.mark_attendance("D3", 90, now=t0 + timedelta(hours=8))["status"], "check_out")
        row = [r for r in db.attendance_between(t0.date().isoformat(), t0.date().isoformat())
               if r["emp_id"] == "D3"][0]
        self.assertTrue(row["late"])                    # 10:00 > 09:30
        self.assertEqual(row["check_out"], (t0 + timedelta(hours=8)).strftime("%H:%M:%S"))
        self.assertIn("D3", db.export_csv(t0.date().isoformat(), t0.date().isoformat()))
        self.assertEqual(db.mark_attendance("NOPE")["status"], "unknown_employee")

    def test_tracker_needs_consecutive_sightings(self):
        from recognizer import Tracker
        tr, hit = Tracker(), [{"emp_id": "X", "confidence": 90}]
        self.assertEqual(tr.update(hit, now=100.0), [])
        self.assertEqual(tr.update(hit, now=100.6), [])
        self.assertEqual(tr.update(hit, now=101.2), ["X"])      # 3rd consecutive
        self.assertEqual(tr.update(hit, now=101.8), [])         # cooldown
        tr2 = Tracker()
        tr2.update(hit, now=0), tr2.update(hit, now=1)
        self.assertEqual(tr2.update(hit, now=20), [])           # gap resets the streak


@unittest.skipIf(FACE is None, "scikit-image not installed")
class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        db.add_employee("R1", "Eileen")
        import recognizer
        importlib.reload(recognizer)
        cls.engine = recognizer.FaceEngine()
        for s in range(15):
            ok, msg = cls.engine.add_sample("R1", variant(FACE, s))
            assert ok, msg
        cls.engine.train()

    def test_registered_face_is_recognised(self):
        hits = 0
        for s in range(100, 110):
            res = self.engine.identify(variant(FACE, s))
            hits += bool(res) and res[0]["emp_id"] == "R1"
        self.assertGreaterEqual(hits, 9, f"only {hits}/10 recognised")

    def test_no_face_means_no_result_and_enroll_rejects(self):
        blank = np.full((480, 640, 3), 127, np.uint8)
        self.assertEqual(self.engine.identify(blank), [])
        ok, _ = self.engine.add_sample("R1", blank)
        self.assertFalse(ok)

    def test_unknown_face_rejected_when_threshold_strict(self):
        # A very strict threshold must reject even the enrolled face -> proves
        # the threshold gate actually controls acceptance.
        old, config.LBPH_THRESHOLD = config.LBPH_THRESHOLD, 0.5
        try:
            res = self.engine.identify(variant(FACE, 7))
            self.assertTrue(res and res[0]["emp_id"] is None)
        finally:
            config.LBPH_THRESHOLD = old


@unittest.skipIf(FACE is None, "scikit-image not installed")
class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.delete_employee("R1")      # same photo must not exist under two identities
        import app as appmod
        importlib.reload(appmod)
        cls.app = appmod.app
        cls.c = appmod.app.test_client()

    def test_full_flow(self):
        c = self.c
        for page in ("/", "/employees", "/kiosk", "/report"):
            self.assertEqual(c.get(page).status_code, 200, page)
        r = c.post("/employees/add", data={"emp_id": "W1", "name": "Web User", "department": "Eng"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(c.get("/employees/W1/enroll").status_code, 200)
        for s in range(15):
            j = c.post("/api/enroll/W1", json={"image": data_url(variant(FACE, s))}).get_json()
            self.assertTrue(j["ok"], j)
        self.assertEqual(j["count"], 15)
        self.assertTrue(c.post("/api/train").get_json()["ok"])

        events = []
        for s in range(200, 204):                      # 4 "frames" of the kiosk
            j = c.post("/api/recognize", json={"image": data_url(variant(FACE, s))}).get_json()
            self.assertEqual(j["faces"][0]["name"], "Web User")
            events += j["events"]
        self.assertEqual([e["status"] for e in events], ["check_in"])   # exactly once

        self.assertIn("Web User", c.get("/").get_data(as_text=True))
        csv_text = c.get("/export.csv").get_data(as_text=True)
        self.assertIn("W1", csv_text)
        self.assertEqual(c.post("/api/recognize", json={"image": "garbage"}).status_code, 400)
        self.assertEqual(c.post("/employees/W1/delete").status_code, 302)
        self.assertIsNone(db.get_employee("W1"))


if __name__ == "__main__":
    unittest.main()
