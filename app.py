"""Web dashboard + browser-camera kiosk.   Run:  python app.py   ->  http://127.0.0.1:5000"""
import base64
import os
from datetime import date

import cv2
import numpy as np
from flask import (Flask, Response, abort, flash, jsonify, redirect,
                   render_template, request, url_for)
from werkzeug.exceptions import HTTPException

import config
import database as db
from recognizer import FaceEngine, Tracker

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

db.init_db()
engine = FaceEngine()
tracker = Tracker()


def decode_image(data_url):
    try:
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.errorhandler(Exception)
def on_error(e):
    """API calls always get JSON back (never an HTML error page)."""
    if isinstance(e, HTTPException):
        return e
    if request.path.startswith("/api/"):
        app.logger.exception("API error")
        return jsonify(ok=False, faces=[], events=[], message=f"{type(e).__name__}: {e}"), 500
    raise e


@app.context_processor
def inject():
    return {"backend": engine.backend, "samples_required": config.SAMPLES_REQUIRED}


# ------------------------------------------------------------------ pages
@app.route("/")
def dashboard():
    return render_template("dashboard.html", stats=db.today_stats(),
                           absent=db.absent_on(date.today().isoformat()),
                           today=date.today().isoformat())


@app.route("/employees")
def employees():
    emps = [dict(e, samples=engine.sample_count(e["emp_id"])) for e in db.list_employees()]
    return render_template("employees.html", employees=emps)


@app.route("/employees/add", methods=["POST"])
def add_employee():
    f = request.form
    try:
        db.add_employee(f.get("emp_id", ""), f.get("name", ""), f.get("department", ""))
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("employees"))
    return redirect(url_for("enroll", emp_id=f["emp_id"].strip()))


@app.route("/employees/<emp_id>/delete", methods=["POST"])
def delete_employee(emp_id):
    db.delete_employee(emp_id)
    engine.train()
    flash("Employee removed.", "ok")
    return redirect(url_for("employees"))


@app.route("/employees/<emp_id>/enroll")
def enroll(emp_id):
    emp = db.get_employee(emp_id)
    if emp is None:
        abort(404)
    return render_template("enroll.html", emp=emp, count=engine.sample_count(emp_id))


@app.route("/kiosk")
def kiosk():
    return render_template("kiosk.html")


@app.route("/report")
def report():
    today = date.today().isoformat()
    start, end = request.args.get("start", today), request.args.get("end", today)
    return render_template("report.html", rows=db.attendance_between(start, end),
                           start=start, end=end)


@app.route("/export.csv")
def export():
    today = date.today().isoformat()
    start, end = request.args.get("start", today), request.args.get("end", today)
    return Response(db.export_csv(start, end), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=attendance_{start}_to_{end}.csv"})


# -------------------------------------------------------------------- API
@app.route("/api/enroll/<emp_id>", methods=["POST"])
def api_enroll(emp_id):
    if db.get_employee(emp_id) is None:
        return jsonify(ok=False, message="Unknown employee"), 404
    img = decode_image((request.get_json(silent=True) or {}).get("image", ""))
    ok, msg = engine.add_sample(emp_id, img)
    return jsonify(ok=ok, message=msg, count=engine.sample_count(emp_id))


@app.route("/api/enroll/<emp_id>/upload", methods=["POST"])
def api_upload(emp_id):
    if db.get_employee(emp_id) is None:
        abort(404)
    saved, problems = 0, []
    for f in request.files.getlist("photos"):
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        ok, msg = engine.add_sample(emp_id, img)
        saved += ok
        if not ok:
            problems.append(f"{f.filename}: {msg}")
    engine.train()
    flash(f"{saved} photo(s) saved." + (" Skipped: " + "; ".join(problems) if problems else ""),
          "ok" if saved else "error")
    return redirect(url_for("enroll", emp_id=emp_id))


@app.route("/api/train", methods=["POST"])
def api_train():
    engine.train()
    return jsonify(ok=True)


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    img = decode_image((request.get_json(silent=True) or {}).get("image", ""))
    if img is None:
        return jsonify(faces=[], events=[]), 400
    results = engine.identify(img)
    names = {e["emp_id"]: e["name"] for e in db.list_employees()}
    events = []
    for emp_id in tracker.update(results):
        conf = max(r["confidence"] for r in results if r["emp_id"] == emp_id)
        res = db.mark_attendance(emp_id, conf)
        if res["status"] != "unknown_employee":
            events.append(res)
    faces = [{"box": r["box"], "name": names.get(r["emp_id"], "Unknown"),
              "known": r["emp_id"] is not None, "confidence": r["confidence"]} for r in results]
    return jsonify(faces=faces, events=events, size=[img.shape[1], img.shape[0]])


if __name__ == "__main__":
    print(f"Backend: {engine.backend}  |  open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)