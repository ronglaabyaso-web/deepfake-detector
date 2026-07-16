# -*- coding: utf-8 -*-
"""Pipeline วิเคราะห์วิดีโอ: ดาวน์โหลด → ตัดเฟรม → ตรวจหน้า → ensemble โมเดล → รวมผล

หลักการ (อิงงานวิจัย 2026: ไม่มีโมเดลเดี่ยวแม่นสุด ควรผสมหลายตัว + เน้นใบหน้า):
1. Ensemble หลายโมเดล — env DEEPFAKE_MODEL_ID รับหลายตัวคั่นด้วย comma
   โหลดทุกตัว (ตัวที่โหลดไม่ได้ให้ข้าม) ต่อเฟรมเฉลี่ยคะแนน "ความเป็นของปลอม" จากทุกโมเดล
2. วิเคราะห์เฉพาะเฟรมที่เจอใบหน้า + ครอปหน้าแบบมีขอบ (FACE_MARGIN ~20%)
   ถ้าไม่เจอใบหน้าเลยทั้งคลิป → ตอบตรง ๆ ว่า "ไม่พบใบหน้า ประเมินไม่ได้" (ไม่เดามั่ว)
3. รวมผลแบบค่าเฉลี่ย + timestamp ที่น่าสงสัย + degrade เมื่อ error + ส่ง models_used
4. MODEL_ID (สตริงรวม) ให้ /api/health แสดงได้

ต้องติดตั้ง requirements-video.txt ก่อน (torch, transformers, opencv, yt-dlp)
ถ้ายังไม่ได้ติดตั้ง — analyze_video() คืน status "unavailable" พร้อมคำอธิบายไทย

ผลลัพธ์ analyze_video(url) มีคีย์: status, risk_percent, frames_analyzed,
faces_found, suspicious_timestamps, error_th (+ models_used, disclaimer_th)
"""
import os
import tempfile

# สตริงรวมของทุกโมเดล — ให้ /api/health แสดง
MODEL_ID = os.environ.get(
    "DEEPFAKE_MODEL_ID",
    "prithivMLmods/deepfake-detector-model-v1,prithivMLmods/Deepfake-Detect-Siglip2",
)
FACE_MARGIN = float(os.environ.get("FACE_MARGIN", "0.2"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "16"))
MAX_VIDEO_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", "180"))

DISCLAIMER_TH = (
    "ผลนี้เป็นเพียงสัญญาณเตือนให้ระวัง ไม่ใช่คำตัดสิน 100% — "
    "เครื่องมือตรวจคลิปปลอมอาจพลาดกับคลิปแบบใหม่ ๆ ได้ "
    "สิ่งที่เชื่อถือได้กว่าคือกฎทอง 3 ข้อ: เร่งรีบ=อันตราย, "
    "วางสายแล้วโทรกลับเบอร์จริงเอง, ไม่โอนเงิน ไม่บอกรหัส ไม่ว่ากรณีใด"
)

# ชื่อ label ที่นับเป็น "ของปลอม" (รองรับหลายแบบตามโมเดลแต่ละตัว)
FAKE_LABEL_HINTS = ("fake", "deepfake", "spoof", "synthetic", "generated", "manipulated", "ai")
REAL_LABEL_HINTS = ("real", "authentic", "genuine", "live", "original", "human")

_models = None            # list ของ (name, processor, model, fake_index)
_deps_available = None    # None = ยังไม่เช็ก, True/False = เช็กแล้ว


def _check_deps() -> bool:
    """เช็กว่าไลบรารีตัวหนักติดตั้งครบไหม (เช็กครั้งเดียว)"""
    global _deps_available
    if _deps_available is None:
        try:
            import cv2            # noqa: F401
            import torch          # noqa: F401
            import transformers   # noqa: F401
            import yt_dlp         # noqa: F401
            _deps_available = True
        except ImportError:
            _deps_available = False
    return _deps_available


def _find_fake_index(id2label: dict) -> int | None:
    """หา index ของ label "ของปลอม" อย่างยืดหยุ่น (รองรับชื่อหลายแบบ / binary real-fake)"""
    labels = {int(i): str(lbl).lower() for i, lbl in id2label.items()}
    for i, lbl in labels.items():
        if any(h in lbl for h in FAKE_LABEL_HINTS):
            return i
    # ไม่เจอชื่อ fake ตรง ๆ — ถ้าเป็น binary และอีกฝั่งคือ real ให้เอาฝั่งตรงข้าม
    if len(labels) == 2:
        for i, lbl in labels.items():
            if any(h in lbl for h in REAL_LABEL_HINTS):
                return next(j for j in labels if j != i)
    return None


def _load_models():
    """โหลดทุกโมเดลจาก DEEPFAKE_MODEL_ID (คั่น comma) — ตัวที่โหลดไม่ได้ให้ข้าม"""
    global _models
    if _models is not None:
        return _models
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    loaded = []
    for model_id in [m.strip() for m in MODEL_ID.split(",") if m.strip()]:
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)
            model.eval()
            fake_idx = _find_fake_index(model.config.id2label)
            if fake_idx is None:
                continue  # ไม่รู้ว่า label ไหนคือของปลอม — ข้ามเพื่อไม่ให้ผลเพี้ยน
            loaded.append((model_id, processor, model, fake_idx))
        except Exception:
            continue  # โหลดไม่ได้ (เน็ตล่ม/ชื่อผิด) — ข้าม ใช้ตัวที่เหลือ
    _models = loaded
    return _models


def _download_video(url: str, workdir: str) -> str:
    """ดาวน์โหลดวิดีโอด้วย yt-dlp คืน path ไฟล์ (raise เมื่อพลาด)"""
    import yt_dlp
    out_tmpl = os.path.join(workdir, "video.%(ext)s")
    opts = {
        "outtmpl": out_tmpl,
        "format": "best[height<=480]/bestvideo[height<=480]+bestaudio/best",
        "max_filesize": 200 * 1024 * 1024,
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration <= {MAX_VIDEO_SECONDS} | duration = None"
        ),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    for name in os.listdir(workdir):
        if name.startswith("video."):
            return os.path.join(workdir, name)
    raise RuntimeError("download produced no file")


def _extract_frames(video_path: str):
    """ตัดเฟรมกระจายทั่วคลิปสูงสุด MAX_FRAMES เฟรม คืน list ของ (timestamp_sec, frame_bgr)"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total <= 0 or fps <= 0:
            return []
        step = max(1, total // MAX_FRAMES)
        frames = []
        for idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append((idx / fps, frame))
            if len(frames) >= MAX_FRAMES:
                break
        return frames
    finally:
        cap.release()


_face_cascade = None


def _detect_face_crop(frame_bgr):
    """ตรวจหาใบหน้าในเฟรม คืนภาพครอปหน้า (มีขอบ FACE_MARGIN) หรือ None ถ้าไม่เจอ"""
    global _face_cascade
    import cv2
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # เอาหน้าที่ใหญ่สุด (ตัวหลักของคลิป)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    return frame_bgr[y0:y1, x0:x1]


def _score_face(face_bgr, models) -> float:
    """ให้คะแนน "ความเป็นของปลอม" 0-1 โดยเฉลี่ยจากทุกโมเดล (ensemble)"""
    import cv2
    import torch
    from PIL import Image
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    scores = []
    for _name, processor, model, fake_idx in models:
        try:
            inputs = processor(images=pil, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            scores.append(float(probs[fake_idx]))
        except Exception:
            continue  # โมเดลตัวนี้พลาดกับเฟรมนี้ — ใช้ตัวที่เหลือ
    if not scores:
        raise RuntimeError("all models failed on frame")
    return sum(scores) / len(scores)


def analyze_video(url: str) -> dict:
    """วิเคราะห์วิดีโอจากลิงก์ คืน dict ตามสัญญาเดิม + degrade อย่างนุ่มนวลเมื่อ error

    คีย์: status, risk_percent, frames_analyzed, faces_found,
          suspicious_timestamps, error_th, models_used, disclaimer_th
    """
    base = {
        "status": "error",
        "risk_percent": None,
        "frames_analyzed": 0,
        "faces_found": 0,
        "suspicious_timestamps": [],
        "error_th": None,
        "models_used": [],
        "disclaimer_th": DISCLAIMER_TH,
    }

    if not _check_deps():
        base["status"] = "unavailable"
        base["error_th"] = (
            "เครื่องนี้ยังไม่ได้ติดตั้งชุดตรวจคลิปวิดีโอ จึงตรวจคลิปไม่ได้ตอนนี้ "
            "แต่ยังตรวจข้อความและถามผู้ช่วยได้ตามปกติ — "
            "ระหว่างนี้ให้ใช้กฎทอง 3 ข้อช่วยสังเกตแทน"
        )
        return base

    models = _load_models()
    base["models_used"] = [name for name, *_ in models]
    if not models:
        base["status"] = "unavailable"
        base["error_th"] = (
            "โหลดตัวตรวจคลิปไม่สำเร็จ (อาจเป็นที่อินเทอร์เน็ต) ลองใหม่อีกครั้งภายหลัง "
            "ระหว่างนี้ให้ใช้กฎทอง 3 ข้อช่วยสังเกตแทน"
        )
        return base

    with tempfile.TemporaryDirectory(prefix="dfvideo_") as workdir:
        # 1) ดาวน์โหลด
        try:
            video_path = _download_video(url, workdir)
        except Exception:
            base["error_th"] = (
                "ดาวน์โหลดคลิปจากลิงก์นี้ไม่ได้ ลองตรวจดูว่าลิงก์ถูกต้อง เปิดดูได้จริง "
                "และคลิปไม่ยาวเกิน 3 นาที แล้วลองใหม่อีกครั้ง"
            )
            return base

        # 2) ตัดเฟรม
        try:
            frames = _extract_frames(video_path)
        except Exception:
            frames = []
        if not frames:
            base["error_th"] = "อ่านภาพจากคลิปนี้ไม่ได้ คลิปอาจเสียหรือเป็นรูปแบบที่ไม่รองรับ"
            return base

        # 3) ตรวจหน้า + 4) ให้คะแนน ensemble เฉพาะเฟรมที่เจอใบหน้า
        per_frame = []  # (timestamp, score)
        faces_found = 0
        for ts, frame in frames:
            face = _detect_face_crop(frame)
            if face is None or face.size == 0:
                continue
            faces_found += 1
            try:
                per_frame.append((ts, _score_face(face, models)))
            except Exception:
                continue  # เฟรมนี้ประเมินพลาด — ข้าม (degrade นุ่มนวล)

        base["frames_analyzed"] = len(frames)
        base["faces_found"] = faces_found

        # ไม่เจอใบหน้าเลยทั้งคลิป → ตอบตรง ๆ ไม่เดามั่ว
        if faces_found == 0:
            base["status"] = "no_face"
            base["error_th"] = (
                "ไม่พบใบหน้าคนในคลิปนี้ จึงประเมินไม่ได้ — "
                "เครื่องมือนี้ตรวจได้เฉพาะคลิปที่เห็นใบหน้าชัด ๆ "
                "ให้ใช้กฎทอง 3 ข้อช่วยตัดสินใจแทน"
            )
            return base

        if not per_frame:
            base["error_th"] = (
                "พบใบหน้าในคลิปแต่ประเมินไม่สำเร็จ ลองใหม่อีกครั้งภายหลัง "
                "ระหว่างนี้ให้ใช้กฎทอง 3 ข้อช่วยสังเกตแทน"
            )
            return base

        # 5) รวมผลแบบค่าเฉลี่ย + timestamp ที่น่าสงสัย (คะแนน > 0.6)
        avg = sum(s for _, s in per_frame) / len(per_frame)
        base["status"] = "ok"
        base["risk_percent"] = round(min(95.0, avg * 100), 1)
        base["suspicious_timestamps"] = [
            round(ts, 1) for ts, s in per_frame if s > 0.6
        ]
        return base
