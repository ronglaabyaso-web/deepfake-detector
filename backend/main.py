# -*- coding: utf-8 -*-
"""ตัวช่วยจับคลิปปลอม — เซิร์ฟเวอร์หลัก (FastAPI)

เซิร์ฟเวอร์เดียวเสิร์ฟทั้ง API และหน้าเว็บ:
- GET  /api/health          สถานะเซิร์ฟเวอร์ + โหมด AI + โมเดลวิดีโอ
- GET  /api/scam-patterns   9 รูปแบบมิจฉาชีพ
- GET  /api/golden-rules    กฎทอง 3 ข้อ
- POST /api/analyze-text    วิเคราะห์ข้อความ/SMS
- POST /api/analyze-video   วิเคราะห์คลิปจากลิงก์
- POST /api/chat            แชทกับผู้ช่วย
- POST /api/register        สมัครสมาชิก
- POST /api/login           เข้าสู่ระบบ
- POST /api/me              ดูข้อมูลตัวเอง (ส่ง token)
- POST /api/logout          ออกจากระบบ
- GET  /                    หน้าเว็บ (frontend/)
"""
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()  # ต้องมาก่อน import โมดูลที่อ่าน env ตอน import

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ai_client
import auth
import scam_data
import video_pipeline

app = FastAPI(title="ตัวช่วยจับคลิปปลอม", docs_url=None, redoc_url=None)

# อนุญาตทุก origin — frontend อาจ deploy แยก (GitHub Pages ฯลฯ) แล้วเรียก backend ผ่าน tunnel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

# วิเคราะห์วิดีโอเป็นงานหนัก (CPU) — จำกัดทีละ 1 งาน กันเครื่องอืด
_video_executor = ThreadPoolExecutor(max_workers=1)


@app.on_event("startup")
def _startup():
    auth.init_db()


# ---------------------------------------------------------------------------
# รูปแบบ request
# ---------------------------------------------------------------------------
class TextIn(BaseModel):
    text: str


class VideoIn(BaseModel):
    url: str


class ChatIn(BaseModel):
    message: str
    history: list[dict] | None = None


class RegisterIn(BaseModel):
    username: str
    password: str


class LoginIn(BaseModel):
    username: str
    password: str


class TokenIn(BaseModel):
    token: str | None = None


# ---------------------------------------------------------------------------
# ข้อมูลพื้นฐาน
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    client = ai_client.get_client()
    return {
        "status": "ok",
        "ai_provider": client.provider,          # gemini/claude/ollama/none
        "video_model_id": video_pipeline.MODEL_ID,
        "video_ready": video_pipeline.video_ready(),
    }


@app.get("/api/scam-patterns")
def scam_patterns():
    # ไม่ส่ง keywords ให้ frontend (เป็นรายละเอียดภายใน)
    patterns = [
        {k: p[k] for k in ("id", "icon", "name_th", "description_th", "warning_th")}
        for p in scam_data.SCAM_PATTERNS
    ]
    return {"patterns": patterns, "hotlines": scam_data.HOTLINES}


@app.get("/api/golden-rules")
def golden_rules():
    return {"rules": scam_data.GOLDEN_RULES}


# ---------------------------------------------------------------------------
# วิเคราะห์ / แชท
# ---------------------------------------------------------------------------
@app.post("/api/analyze-text")
def analyze_text(body: TextIn):
    text = (body.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error_th": "กรุณาใส่ข้อความที่ต้องการตรวจก่อนนะคะ"})
    if len(text) > 5000:
        text = text[:5000]
    return ai_client.get_client().analyze_text(text)


@app.post("/api/analyze-video")
async def analyze_video(body: VideoIn):
    url = (body.url or "").strip()
    if not url or not url.lower().startswith(("http://", "https://")):
        return JSONResponse(
            status_code=400,
            content={"error_th": "กรุณาวางลิงก์คลิปที่ขึ้นต้นด้วย http:// หรือ https:// นะคะ"},
        )
    # งานหนัก — รันใน thread แยก ไม่บล็อกผู้ใช้คนอื่น
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_video_executor, video_pipeline.analyze_video, url)


@app.post("/api/chat")
def chat(body: ChatIn):
    message = (body.message or "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"error_th": "พิมพ์คำถามก่อนนะคะ"})
    return ai_client.get_client().chat(message, body.history)


# ---------------------------------------------------------------------------
# ระบบสมาชิก
# ---------------------------------------------------------------------------
@app.post("/api/register")
def register(body: RegisterIn):
    result = auth.register(body.username, body.password)
    if not result["ok"]:
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/api/login")
def login(body: LoginIn):
    result = auth.login(body.username, body.password)
    if not result["ok"]:
        return JSONResponse(status_code=401, content=result)
    return result


@app.post("/api/me")
def me(body: TokenIn):
    username = auth.get_user(body.token)
    if username is None:
        return JSONResponse(
            status_code=401,
            content={"ok": False, "error_th": "หมดเวลาเข้าสู่ระบบ กรุณาเข้าสู่ระบบใหม่นะคะ"},
        )
    return {"ok": True, "username": username}


@app.post("/api/logout")
def logout(body: TokenIn):
    auth.logout(body.token)
    return {"ok": True}


# ---------------------------------------------------------------------------
# เสิร์ฟ frontend (ต้อง mount ท้ายสุด ไม่งั้นทับ /api)
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
