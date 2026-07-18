# -*- coding: utf-8 -*-
"""Pipeline วิเคราะห์วิดีโอ: ดาวน์โหลด → ตัดเฟรม → ตรวจหน้า → ensemble โมเดล → รวมผล

หลักการ (อิงงานวิจัย 2026: ไม่มีโมเดลเดี่ยวแม่นสุด ควรผสมหลายตัว + เน้นใบหน้า):
1. Ensemble หลายโมเดล — env DEEPFAKE_MODEL_ID รับหลายตัวคั่นด้วย comma
   เฉลี่ยคะแนน "ความเป็นของปลอม" จากทุกโมเดลต่อเฟรม (ตัวที่ใช้ไม่ได้ให้ข้าม)
2. วิเคราะห์เฉพาะเฟรมที่เจอใบหน้า + ครอปหน้าแบบมีขอบ (FACE_MARGIN ~20%)
   ถ้าไม่เจอใบหน้าเลยทั้งคลิป → ตอบตรง ๆ ว่า "ไม่พบใบหน้า ประเมินไม่ได้" (ไม่เดามั่ว)
3. รวมผลแบบค่าเฉลี่ย + timestamp ที่น่าสงสัย + degrade เมื่อ error + ส่ง models_used
4. MODEL_ID (สตริงรวม) ให้ /api/health แสดงได้

สองโหมดการให้คะแนน (เลือกอัตโนมัติ):
- local  : ติดตั้ง torch+transformers → โหลดโมเดลรันในเครื่อง (แม่น/เป็นส่วนตัว แต่กินแรม)
- remote : ตั้ง env DEEPFAKE_HF_TOKEN → เรียก Hugging Face Inference API แทน
           (เบามาก รันบนเซิร์ฟเวอร์แรมน้อยได้ เช่น 512MB — ต้องมีแค่ opencv + yt-dlp)

ทั้งสองโหมดต้องมี opencv (cv2) + yt-dlp สำหรับดาวน์โหลด/ตัดเฟรม/ตรวจหน้า
ถ้าขาด — analyze_video() คืน status "unavailable" พร้อมคำอธิบายไทย

ผลลัพธ์ analyze_video(url) มีคีย์: status, risk_percent, frames_analyzed,
faces_found, suspicious_timestamps, error_th (+ models_used, disclaimer_th)
"""
import os
import tempfile
import time

# สตริงรวมของทุกโมเดล — ให้ /api/health แสดง
MODEL_ID = os.environ.get(
    "DEEPFAKE_MODEL_ID",
    "prithivMLmods/deepfake-detector-model-v1,prithivMLmods/Deepfake-Detect-Siglip2",
)
FACE_MARGIN = float(os.environ.get("FACE_MARGIN", "0.2"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "16"))
MAX_VIDEO_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", "180"))

# โหมด remote (Hugging Face Inference API) — ตั้ง token ฝั่งเซิร์ฟเวอร์เท่านั้น
HF_TOKEN = os.environ.get("DEEPFAKE_HF_TOKEN", "").strip()
HF_ROUTER = "https://router.huggingface.co/hf-inference/models/"
# remote เรียก API ทีละเฟรม จึงจำกัดจำนวนหน้าที่ส่งไปเพื่อความเร็ว/โควตา
REMOTE_MAX_FACES = int(os.environ.get("REMOTE_MAX_FACES", "8"))

DISCLAIMER_TH = (
    "ผลนี้เป็นเพียงสัญญาณเตือนให้ระวัง ไม่ใช่คำตัดสิน 100% — "
    "เครื่องมือตรวจคลิปปลอมอาจพลาดกับคลิปแบบใหม่ ๆ ได้ "
    "สิ่งที่เชื่อถือได้กว่าคือกฎทอง 3 ข้อ: เร่งรีบ=อันตราย, "
    "วางสายแล้วโทรกลับเบอร์จริงเอง, ไม่โอนเงิน ไม่บอกรหัส ไม่ว่ากรณีใด"
)

# ชื่อ label ที่นับเป็น "ของปลอม" (รองรับหลายแบบตามโมเดลแต่ละตัว)
FAKE_LABEL_HINTS = ("fake", "deepfake", "spoof", "synthetic", "generated", "manipulated", "ai")
REAL_LABEL_HINTS = ("real", "authentic", "genuine", "live", "original", "human")

_models = None            # โหมด local: list ของ (name, processor, model, fake_index)
_remote_models = None      # โหมด remote: list ของ model_id ที่ยังใช้งานได้
_video_deps = None         # None = ยังไม่เช็ก, True/False = เช็กแล้ว (cv2 + yt_dlp)
_local_ready = None        # torch + transformers ติดตั้งไหม


def _has_video_deps() -> bool:
    """เช็กไลบรารีดาวน์โหลด/ตัดเฟรม/ตรวจหน้า (cv2 + yt_dlp) — เช็กครั้งเดียว"""
    global _video_deps
    if _video_deps is None:
        try:
            import cv2       # noqa: F401
            import yt_dlp    # noqa: F401
            from PIL import Image  # noqa: F401
            _video_deps = True
        except ImportError:
            _video_deps = False
    return _video_deps


def _has_local_models() -> bool:
    """เช็กว่ารันโมเดลในเครื่องได้ไหม (torch + transformers)"""
    global _local_ready
    if _local_ready is None:
        try:
            import torch          # noqa: F401
            import transformers   # noqa: F401
            _local_ready = True
        except ImportError:
            _local_ready = False
    return _local_ready


def _mode() -> str | None:
    """เลือกโหมดให้คะแนน: 'local' (torch) > 'remote' (HF API) > None (ทำไม่ได้)"""
    if not _has_video_deps():
        return None
    if _has_local_models():
        return "local"
    if HF_TOKEN:
        return "remote"
    return None


def video_ready() -> bool:
    """ตรวจวิดีโอได้ไหม (ให้ /api/health ใช้) — ต้องมี deps + ช่องทางให้คะแนน"""
    return _mode() is not None


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


def _fake_score_from_labels(results) -> float | None:
    """แปลงผล [{'label':..,'score':..}, ...] เป็นคะแนน "ความเป็นของปลอม" 0-1"""
    if not isinstance(results, list) or not results:
        return None
    fake, real = None, None
    for item in results:
        lbl = str(item.get("label", "")).lower()
        sc = float(item.get("score", 0))
        if any(h in lbl for h in FAKE_LABEL_HINTS):
            fake = sc
        elif any(h in lbl for h in REAL_LABEL_HINTS):
            real = sc
    if fake is not None:
        return fake
    if real is not None:  # มีแต่ label real → ของปลอม = ส่วนที่เหลือ
        return 1.0 - real
    return None


# ---------------------------------------------------------------------------
# โหมด local (torch)
# ---------------------------------------------------------------------------
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


def _score_face_local(face_bgr, models) -> float:
    """ให้คะแนน "ความเป็นของปลอม" 0-1 โดยเฉลี่ยจากทุกโมเดลในเครื่อง (ensemble)"""
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


# ---------------------------------------------------------------------------
# โหมด remote (Hugging Face Inference API) — ไม่ต้องมี torch
# ---------------------------------------------------------------------------
def _init_remote_models():
    """คืน list ของ model_id ทั้งหมดจาก DEEPFAKE_MODEL_ID (ตัวที่ใช้ไม่ได้จะถูกตัดตอนเรียกจริง)"""
    global _remote_models
    if _remote_models is None:
        _remote_models = [m.strip() for m in MODEL_ID.split(",") if m.strip()]
    return _remote_models


def _encode_jpeg(face_bgr) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _score_face_remote(face_bgr, model_ids) -> float:
    """ให้คะแนน 0-1 โดยเฉลี่ยจากทุกโมเดลผ่าน HF Inference API (ตัดโมเดลที่ตายทิ้ง)"""
    import httpx
    jpg = _encode_jpeg(face_bgr)
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "image/jpeg"}
    scores = []
    dead = []
    for model_id in model_ids:
        score = None
        for attempt in range(3):
            try:
                r = httpx.post(HF_ROUTER + model_id, headers=headers, content=jpg, timeout=45)
            except Exception:
                break  # เน็ตพลาด — ข้ามโมเดลนี้เฟรมนี้
            if r.status_code == 503:      # โมเดลกำลัง warm up — รอแล้วลองใหม่
                time.sleep(6)
                continue
            if r.status_code in (404, 410, 400):
                dead.append(model_id)     # โมเดลถูกเลิกซัพพอร์ต — เลิกเรียกถาวร
                break
            if r.status_code == 200:
                score = _fake_score_from_labels(r.json())
            break
        if score is not None:
            scores.append(score)
    # ตัดโมเดลที่ตายออกจากรายการ (ไม่เรียกซ้ำเฟรมต่อไป)
    if dead:
        model_ids[:] = [m for m in model_ids if m not in dead]
    if not scores:
        raise RuntimeError("all remote models failed on frame")
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# ดาวน์โหลด / ตัดเฟรม / ตรวจหน้า (ใช้ร่วมกันทั้งสองโหมด)
# ---------------------------------------------------------------------------
def _download_video(url: str, workdir: str) -> str:
    """ดาวน์โหลดวิดีโอด้วย yt-dlp คืน path ไฟล์ (raise เมื่อพลาด)

    เลือกฟอร์แมตไฟล์เดียว (progressive) ก่อน เพื่อไม่ต้องพึ่ง ffmpeg บนเซิร์ฟเวอร์เล็ก
    """
    import yt_dlp
    out_tmpl = os.path.join(workdir, "video.%(ext)s")

    def _too_long(info, *, incomplete=False):
        dur = info.get("duration")
        if dur is not None and dur > MAX_VIDEO_SECONDS:
            return f"คลิปยาวเกิน {MAX_VIDEO_SECONDS} วินาที"
        return None  # None = ผ่าน (รวมกรณีไม่รู้ความยาว)

    opts = {
        "outtmpl": out_tmpl,
        "format": "best[ext=mp4][height<=480]/best[ext=mp4]/mp4/best",
        "max_filesize": 200 * 1024 * 1024,
        "match_filter": _too_long,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "http_headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
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

    mode = _mode()
    if mode is None:
        base["status"] = "unavailable"
        if not _has_video_deps():
            base["error_th"] = (
                "เครื่องนี้ยังไม่ได้ติดตั้งชุดตรวจคลิปวิดีโอ จึงตรวจคลิปไม่ได้ตอนนี้ "
                "แต่ยังตรวจข้อความและถามผู้ช่วยได้ตามปกติ — "
                "ระหว่างนี้ให้ใช้กฎทอง 3 ข้อช่วยสังเกตแทน"
            )
        else:
            base["error_th"] = (
                "ยังไม่ได้ตั้งค่าตัวตรวจคลิป จึงตรวจคลิปไม่ได้ตอนนี้ "
                "แต่ยังตรวจข้อความและถามผู้ช่วยได้ตามปกติ"
            )
        return base

    # เตรียมรายการโมเดลตามโหมด
    if mode == "local":
        models = _load_models()
        base["models_used"] = [name for name, *_ in models]
        if not models:
            base["status"] = "unavailable"
            base["error_th"] = (
                "โหลดตัวตรวจคลิปไม่สำเร็จ (อาจเป็นที่อินเทอร์เน็ต) ลองใหม่อีกครั้งภายหลัง "
                "ระหว่างนี้ให้ใช้กฎทอง 3 ข้อช่วยสังเกตแทน"
            )
            return base
    else:  # remote
        models = _init_remote_models()
        base["models_used"] = list(models)

    with tempfile.TemporaryDirectory(prefix="dfvideo_") as workdir:
        # 1) ดาวน์โหลด
        try:
            video_path = _download_video(url, workdir)
        except Exception:
            base["error_th"] = (
                "ดาวน์โหลดคลิปจากลิงก์นี้ไม่ได้ อาจเป็นเพราะแพลตฟอร์มบล็อกการโหลด "
                "หรือคลิปเป็นส่วนตัว/ยาวเกิน 3 นาที — ลองใช้ลิงก์คลิปสาธารณะอื่น "
                "หรือใช้กฎทอง 3 ข้อช่วยสังเกตแทน"
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

        # 3) ตรวจหน้า + 4) ให้คะแนนเฉพาะเฟรมที่เจอใบหน้า
        per_frame = []  # (timestamp, score)
        faces_found = 0
        for ts, frame in frames:
            face = _detect_face_crop(frame)
            if face is None or face.size == 0:
                continue
            faces_found += 1
            # โหมด remote จำกัดจำนวนหน้าที่ส่งไป API เพื่อความเร็ว/โควตา
            if mode == "remote" and len(per_frame) >= REMOTE_MAX_FACES:
                continue
            try:
                if mode == "local":
                    per_frame.append((ts, _score_face_local(face, models)))
                else:
                    per_frame.append((ts, _score_face_remote(face, models)))
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
        # โหมด remote: models_used = เฉพาะตัวที่ยังไม่ตาย
        if mode == "remote":
            base["models_used"] = list(models)
        return base
