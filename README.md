# DEEPFAKE DETECTOR (ตัวช่วยจับคลิปปลอม)

> 🌐 **ใช้งานเลย:** https://deepfake-detector-x4ae.onrender.com
> (โฮสต์ฟรีบน Render — ถ้าไม่มีคนใช้สักพักแอปจะหลับ เข้าครั้งแรกอาจรอ ~1 นาที)

เครื่องมือฟรีสำหรับคนไทย ใช้ตรวจ **คลิป / ลิงก์ / ข้อความ** ที่น่าสงสัยว่าเป็น Deepfake หรือมิจฉาชีพ
ออกแบบสำหรับเด็ก ผู้สูงอายุ และคนต่างจังหวัด — ตัวอักษรใหญ่ ปุ่มกดง่าย ภาษาไทยล้วน

> ผลตรวจทุกอย่างเป็น **"สัญญาณเตือนให้ระวัง" ไม่ใช่คำตัดสิน 100%**
> สิ่งที่เชื่อถือได้กว่าคือ "กฎทอง 3 ข้อ" (สังเกตพฤติกรรมมิจฉาชีพ) ซึ่งแอปสอนควบคู่เสมอ

## ความสามารถ

- 🎬 **ตรวจคลิปวิดีโอ** — วางลิงก์ หรือ **อัปโหลดไฟล์คลิป** วิเคราะห์ใบหน้าด้วยโมเดลตรวจของปลอม
  - บนเวอร์ชันออนไลน์ใช้ **Hugging Face Inference API** ให้คะแนน (ไม่ต้องมี GPU/torch บนเซิร์ฟเวอร์)
  - ลิงก์บางแพลตฟอร์ม (เช่น TikTok/YouTube) มักบล็อกการโหลดจากเซิร์ฟเวอร์คลาวด์ → **แนะนำให้อัปโหลดไฟล์** (เซฟคลิปลงเครื่องก่อน) จะได้ผลชัวร์กว่า
- 💬 **ตรวจข้อความ/SMS** — จับ 9 รูปแบบมิจฉาชีพที่พบบ่อยในไทย
- 🤖 **ผู้ช่วยตอบคำถาม** — แชทถามเรื่องกลโกงออนไลน์ แนะนำสายด่วน 1441 / 191 เมื่อถูกหลอก
- 📚 **เรียนรู้** — กฎทอง 3 ข้อ + รูปแบบมิจฉาชีพ + แบบทดสอบ 7 ข้อ
- 👤 **ระบบสมาชิกในตัว** (ชื่อผู้ใช้+รหัสผ่าน) + โหมดผู้เยี่ยมชม
- 📱 ติดตั้งเป็นแอปบนมือถือได้ (PWA) — ดู [INSTALL_APP.md](INSTALL_APP.md)

## วิธีติดตั้ง / รัน

ต้องมี Python 3.10–3.12

```bash
cd backend
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt     # หลัก (เบา) — พอสำหรับตรวจข้อความ+แชท+เรียนรู้
cp .env.example .env                # (ไม่บังคับ) ใส่ GEMINI_API_KEY ฟรีจาก aistudio.google.com/apikey
python run.py                       # เปิด http://localhost:8000
```

- **ไม่มี key ก็รันได้** — ระบบใช้โหมด "กฎพื้นฐาน" (จับรูปแบบมิจฉาชีพจากฐานข้อมูลในเครื่อง ฟรี 100%)
- อยากได้ตรวจคลิปวิดีโอ: `pip install -r requirements-video.txt` (ตัวหนัก: torch ฯลฯ)
- โมเดลตรวจวิดีโอบน CPU ช้า มี GPU จะเร็วกว่ามาก

## เทคโนโลยี (open source ทั้งหมด, ฟรี 100%)

| ส่วน | เทคโนโลยี |
|---|---|
| Backend | FastAPI (เซิร์ฟเวอร์เดียวเสิร์ฟทั้ง API และหน้าเว็บ) |
| Frontend | PWA ไฟล์เดียว (vanilla JS + HTML + CSS) |
| AI อธิบายผล/แชท | Google Gemini ฟรี (ค่าเริ่มต้น) / Claude / Ollama / โหมดกฎพื้นฐาน |
| ตรวจ Deepfake | โมเดล Hugging Face (ensemble) + OpenCV + yt-dlp รันในเครื่อง |
| ฐานข้อมูลผู้ใช้ | SQLite |

## โครงสร้างโปรเจกต์

```
deepfake-detector/
├── CLAUDE.md                 บริบทโปรเจกต์ (อ่านก่อนเสมอ)
├── README.md                 ไฟล์นี้
├── INSTALL_APP.md            วิธีทำให้เป็นแอปติดตั้งได้ (PWA + deploy ฟรี)
├── backend/
│   ├── main.py               ไฟล์หลัก: endpoint ทั้งหมด + เสิร์ฟ frontend
│   ├── run.py                รันด้วย `python run.py`
│   ├── scam_data.py          9 รูปแบบมิจฉาชีพ + กฎทอง 3 ข้อ + match_patterns()
│   ├── ai_client.py          ตัวเชื่อม AI (gemini/claude/ollama/none)
│   ├── auth.py               ระบบสมาชิก SQLite + hash รหัสผ่าน
│   ├── video_pipeline.py     ดาวน์โหลด+ตัดเฟรม+ตรวจหน้า+โมเดล deepfake
│   ├── requirements.txt      ไลบรารีหลัก (เบา)
│   ├── requirements-video.txt ไลบรารีตรวจวิดีโอ (หนัก)
│   └── .env.example
└── frontend/
    ├── index.html            ทั้งแอปอยู่ในไฟล์นี้
    ├── manifest.json
    ├── sw.js
    └── icon-192.png / icon-512.png / apple-touch-icon.png
```

## API

| Method | Path | ทำอะไร |
|---|---|---|
| GET | `/api/health` | สถานะเซิร์ฟเวอร์ + โหมด AI + โมเดลวิดีโอ |
| GET | `/api/scam-patterns` | 9 รูปแบบมิจฉาชีพ |
| GET | `/api/golden-rules` | กฎทอง 3 ข้อ |
| POST | `/api/analyze-text` | วิเคราะห์ข้อความ/SMS |
| POST | `/api/analyze-video` | วิเคราะห์คลิปจากลิงก์ |
| POST | `/api/chat` | แชทกับผู้ช่วย |
| POST | `/api/register` | สมัครสมาชิก |
| POST | `/api/login` | เข้าสู่ระบบ |
| POST | `/api/me` | ดูข้อมูลตัวเอง (ส่ง token) |
| POST | `/api/logout` | ออกจากระบบ |

## ที่มา

- ใช้แข่ง **Samsung Solve for Tomorrow 2026** (หัวข้อ Cybersecurity)
- ต่อยอดจากโปรเจกต์ที่ชนะ **NCSA Cybersecurity Product and Service Award 2025**
