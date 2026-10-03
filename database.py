"""SQLite storage for employees and attendance."""
import csv
import io
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date

import config

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    emp_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id TEXT NOT NULL REFERENCES employees(emp_id) ON DELETE CASCADE,
    day TEXT NOT NULL,
    check_in TEXT NOT NULL,
    check_out TEXT,
    confidence REAL,
    UNIQUE (emp_id, day)
);
CREATE INDEX IF NOT EXISTS idx_att_day ON attendance(day);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as c:
        c.executescript(SCHEMA)


# ---------------------------------------------------------------- employees
def add_employee(emp_id, name, department=""):
    emp_id, name = emp_id.strip(), name.strip()
    if not ID_RE.match(emp_id):
        raise ValueError("ID may only contain letters, digits, '-' and '_' (max 32).")
    if not name:
        raise ValueError("Name is required.")
    with connect() as c:
        try:
            c.execute("INSERT INTO employees VALUES (?,?,?,?)",
                      (emp_id, name, department.strip(),
                       datetime.now().isoformat(timespec="seconds")))
        except sqlite3.IntegrityError:
            raise ValueError(f"Employee ID '{emp_id}' already exists.")


def get_employee(emp_id):
    with connect() as c:
        return c.execute("SELECT * FROM employees WHERE emp_id=?", (emp_id,)).fetchone()


def list_employees():
    with connect() as c:
        return c.execute("SELECT * FROM employees ORDER BY name").fetchall()


def delete_employee(emp_id):
    with connect() as c:
        c.execute("DELETE FROM attendance WHERE emp_id=?", (emp_id,))
        c.execute("DELETE FROM employees WHERE emp_id=?", (emp_id,))
    if ID_RE.match(emp_id):
        shutil.rmtree(config.DATASET_DIR / emp_id, ignore_errors=True)


# --------------------------------------------------------------- attendance
def mark_attendance(emp_id, confidence=None, now=None):
    """Record a sighting. Returns {'status': 'check_in'|'check_out'|'already', ...}."""
    now = now or datetime.now()
    day, t = now.date().isoformat(), now.strftime("%H:%M:%S")
    with connect() as c:
        emp = c.execute("SELECT name FROM employees WHERE emp_id=?", (emp_id,)).fetchone()
        if emp is None:
            return {"status": "unknown_employee"}
        row = c.execute("SELECT * FROM attendance WHERE emp_id=? AND day=?",
                        (emp_id, day)).fetchone()
        if row is None:
            c.execute("INSERT INTO attendance (emp_id, day, check_in, confidence) "
                      "VALUES (?,?,?,?)", (emp_id, day, t, confidence))
            return {"status": "check_in", "name": emp["name"], "time": t}
        first = datetime.strptime(f"{day} {row['check_in']}", "%Y-%m-%d %H:%M:%S")
        if (now - first).total_seconds() >= config.MIN_CHECKOUT_MINUTES * 60:
            c.execute("UPDATE attendance SET check_out=? WHERE id=?", (t, row["id"]))
            return {"status": "check_out", "name": emp["name"], "time": t}
        return {"status": "already", "name": emp["name"], "time": row["check_in"]}


def _with_late(rows):
    out = []
    for r in rows:
        d = dict(r)
        d["late"] = d["check_in"][:5] > config.WORK_START
        out.append(d)
    return out


def attendance_between(start, end):
    with connect() as c:
        rows = c.execute(
            "SELECT a.day, a.emp_id, e.name, e.department, a.check_in, a.check_out, a.confidence "
            "FROM attendance a JOIN employees e USING(emp_id) "
            "WHERE a.day BETWEEN ? AND ? ORDER BY a.day DESC, a.check_in", (start, end)).fetchall()
    return _with_late(rows)


def absent_on(day):
    with connect() as c:
        return c.execute(
            "SELECT * FROM employees WHERE emp_id NOT IN "
            "(SELECT emp_id FROM attendance WHERE day=?) ORDER BY name", (day,)).fetchall()


def today_stats():
    day = date.today().isoformat()
    rows = attendance_between(day, day)
    total = len(list_employees())
    return {"total": total, "present": len(rows),
            "absent": total - len(rows), "late": sum(r["late"] for r in rows),
            "rows": rows}


def export_csv(start, end):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Date", "Employee ID", "Name", "Department", "Check-in", "Check-out", "Late"])
    for r in attendance_between(start, end):
        w.writerow([r["day"], r["emp_id"], r["name"], r["department"], r["check_in"],
                    r["check_out"] or "", "Yes" if r["late"] else "No"])
    return buf.getvalue()
