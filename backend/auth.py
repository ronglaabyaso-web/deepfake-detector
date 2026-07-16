# -*- coding: utf-8 -*-
"""ระบบสมาชิกในตัว: SQLite + PBKDF2 + session ใน memory

- ไม่ใช้ Google/LINE login — ชื่อผู้ใช้ + รหัสผ่านเท่านั้น
- รหัสผ่าน hash ด้วย PBKDF2-HMAC-SHA256 (600,000 รอบ) + salt สุ่มต่อคน
- session token เก็บใน memory (หายเมื่อรีสตาร์ต — ผู้ใช้ล็อกอินใหม่ได้)
- ข้อความ error ทั้งหมดเป็นภาษาไทยเข้าใจง่าย
"""
import contextlib
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

PBKDF2_ITERATIONS = 600_000
SESSION_LIFETIME_SECONDS = 7 * 24 * 3600  # 7 วัน

# session token -> {"username": ..., "expires": ...}
_sessions: dict[str, dict] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# ฐานข้อมูล
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _connect():
    """เปิด-commit-ปิด ให้ครบ (with sqlite3.connect เฉย ๆ ไม่ปิดไฟล์ ทำให้ค้างบน Windows)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at REAL NOT NULL
            )"""
        )


# ---------------------------------------------------------------------------
# รหัสผ่าน
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)


def _validate_username(username: str) -> str | None:
    """คืนข้อความ error ภาษาไทย หรือ None ถ้าผ่าน"""
    if not username or len(username) < 3:
        return "ชื่อผู้ใช้ต้องยาวอย่างน้อย 3 ตัวอักษร"
    if len(username) > 30:
        return "ชื่อผู้ใช้ต้องไม่เกิน 30 ตัวอักษร"
    if not re.fullmatch(r"[a-zA-Z0-9_฀-๿]+", username):
        return "ชื่อผู้ใช้ใช้ได้เฉพาะตัวอักษรไทย อังกฤษ ตัวเลข และขีดล่าง (_)"
    return None


def _validate_password(password: str) -> str | None:
    if not password or len(password) < 6:
        return "รหัสผ่านต้องยาวอย่างน้อย 6 ตัวอักษร"
    if len(password) > 100:
        return "รหัสผ่านยาวเกินไป"
    return None


# ---------------------------------------------------------------------------
# สมัคร / เข้าสู่ระบบ / session
# ---------------------------------------------------------------------------
def register(username: str, password: str) -> dict:
    """สมัครสมาชิก คืน {'ok': True, 'token': ..., 'username': ...} หรือ {'ok': False, 'error_th': ...}"""
    username = (username or "").strip()
    err = _validate_username(username) or _validate_password(password or "")
    if err:
        return {"ok": False, "error_th": err}
    salt = secrets.token_bytes(16)
    pw_hash = _hash_password(password, salt)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, salt, pw_hash, time.time()),
            )
    except sqlite3.IntegrityError:
        return {"ok": False, "error_th": "ชื่อผู้ใช้นี้มีคนใช้แล้ว ลองชื่ออื่นนะคะ"}
    token = _create_session(username)
    return {"ok": True, "token": token, "username": username}


def login(username: str, password: str) -> dict:
    """เข้าสู่ระบบ คืนรูปแบบเดียวกับ register()"""
    username = (username or "").strip()
    with _connect() as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return {"ok": False, "error_th": "ไม่พบชื่อผู้ใช้นี้ หรือรหัสผ่านไม่ถูกต้อง"}
    expected = row["password_hash"]
    actual = _hash_password(password or "", row["salt"])
    if not hmac.compare_digest(expected, actual):
        return {"ok": False, "error_th": "ไม่พบชื่อผู้ใช้นี้ หรือรหัสผ่านไม่ถูกต้อง"}
    token = _create_session(username)
    return {"ok": True, "token": token, "username": username}


def _create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    with _lock:
        _sessions[token] = {"username": username, "expires": time.time() + SESSION_LIFETIME_SECONDS}
    return token


def get_user(token: str | None) -> str | None:
    """คืน username จาก token หรือ None ถ้า token ใช้ไม่ได้/หมดอายุ"""
    if not token:
        return None
    with _lock:
        sess = _sessions.get(token)
        if sess is None:
            return None
        if sess["expires"] < time.time():
            del _sessions[token]
            return None
        return sess["username"]


def logout(token: str | None) -> bool:
    if not token:
        return False
    with _lock:
        return _sessions.pop(token, None) is not None
