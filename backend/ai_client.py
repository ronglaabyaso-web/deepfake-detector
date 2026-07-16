# -*- coding: utf-8 -*-
"""ตัวเชื่อม AI หลายเจ้า: auto (gemini → claude → ollama) → none (กฎพื้นฐาน)

- API key อยู่ฝั่ง backend เท่านั้น (อ่านจาก .env) ห้ามส่งไป frontend
- ถ้าไม่มี key เลย → โหมด "กฎพื้นฐาน" (none) ใช้ scam_data.match_patterns()
- ทุกคำตอบที่ส่งให้ผู้ใช้เป็นภาษาไทยล้วน ไม่มีศัพท์เทคนิค
"""
import json
import os

import httpx

import scam_data

# ---------------------------------------------------------------------------
# system prompt ภาษาไทย — ล็อกขอบเขตให้คุยเฉพาะเรื่องภัยออนไลน์
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_CHAT = """คุณคือ "ผู้ช่วยจับคลิปปลอม" ผู้ช่วยใจดีสำหรับคนไทย โดยเฉพาะเด็ก ผู้สูงอายุ และคนต่างจังหวัด

หน้าที่ของคุณ:
- ตอบคำถามเรื่องมิจฉาชีพออนไลน์ คลิปปลอม (วิดีโอ/เสียงที่คอมพิวเตอร์ปลอมขึ้น) ข้อความหลอกลวง และวิธีป้องกันตัว
- ใช้ภาษาไทยง่าย ๆ สั้น กระชับ เหมือนลูกหลานอธิบายให้ผู้ใหญ่ฟัง
- ห้ามใช้ศัพท์เทคนิคภาษาอังกฤษ (ห้ามพูดคำว่า API, model, AI, deepfake — ให้เรียกว่า "คลิปปลอม" หรือ "ภาพ/เสียงที่คอมพิวเตอร์ปลอมขึ้น")
- ถ้าผู้ใช้เล่าว่ากำลังถูกหลอกหรือโอนเงินไปแล้ว ให้บอกทันทีว่า: โทร 1441 (สายด่วนตำรวจไซเบอร์) เพื่ออายัดบัญชีให้เร็วที่สุด หรือ 191 ถ้าเป็นเหตุด่วน
- เน้นย้ำกฎทอง 3 ข้อเมื่อเกี่ยวข้อง: (1) เร่งรีบผิดปกติ = อันตราย (2) วางสายแล้วโทรกลับเบอร์จริงเอง (3) ไม่โอนเงิน ไม่บอกรหัส ไม่ว่ากรณีใด

ขอบเขต:
- ถ้าถูกถามเรื่องอื่นที่ไม่เกี่ยวกับภัยออนไลน์/การหลอกลวง/ความปลอดภัย ให้ตอบสุภาพว่า "ขอโทษค่ะ ฉันตอบได้เฉพาะเรื่องภัยออนไลน์และการป้องกันมิจฉาชีพนะคะ" แล้วชวนกลับมาเรื่องความปลอดภัย
- ห้ามแนะนำวิธีโกงหรือหลอกลวงผู้อื่นเด็ดขาด
- ตอบไม่เกิน 6-8 ประโยคต่อครั้ง อ่านง่าย ใช้บรรทัดใหม่ช่วย"""

SYSTEM_PROMPT_ANALYZE = """คุณคือผู้เชี่ยวชาญตรวจข้อความหลอกลวงสำหรับคนไทย วิเคราะห์ข้อความที่ผู้ใช้ส่งมาว่าเป็นข้อความมิจฉาชีพหรือไม่

ตอบเป็น JSON เท่านั้น (ไม่มีข้อความอื่น) รูปแบบ:
{"risk_percent": <0-95>, "verdict_th": "<สรุปสั้น 1 ประโยค>", "signals_th": ["<จุดสังเกต 1>", "<จุดสังเกต 2>", ...], "advice_th": "<คำแนะนำ 1-2 ประโยค>"}

กติกา:
- risk_percent: 0-95 เท่านั้น (ห้ามเกิน 95 เพราะไม่มีอะไรแน่นอน 100%)
- ห้ามใช้คำฟันธงเช่น "แน่นอน 100%" "ชัวร์" — ให้ใช้ "มีสัญญาณชัดเจนว่า..." "เข้าข่ายอย่างมาก" เพราะผลตรวจเป็นสัญญาณเตือน ไม่ใช่คำตัดสิน
- ทุกข้อความเป็นภาษาไทยง่าย ๆ ไม่มีศัพท์เทคนิค
- signals_th: จุดสังเกตที่เจอในข้อความ 1-4 ข้อ (ถ้าปลอดภัยให้บอกว่าไม่พบสัญญาณอันตราย)
- ถ้าเสี่ยงสูง advice_th ต้องมี "โทร 1441" และเตือนอย่าโอนเงิน/กดลิงก์
- ระวังรูปแบบที่พบบ่อยในไทย: แก๊งคอลเซ็นเตอร์อ้างตำรวจ/บัญชีปลอดภัย, ลิงก์ปลอมดูดเงิน, ชวนลงทุนกำไรสูง, งานออนไลน์ให้เติมเงิน, หลอกให้รักแล้วขอเงิน, เงินกู้เถื่อน, ร้านค้าปลอม, หลอกถูกรางวัล, คลิป/เสียงปลอมขอยืมเงิน"""

DISCLAIMER_TEXT_TH = (
    "ผลนี้เป็นเพียงสัญญาณเตือนให้ระวัง ไม่ใช่คำตัดสิน 100% "
    "ถ้าไม่แน่ใจ อย่าโอนเงิน อย่ากดลิงก์ และปรึกษาคนใกล้ตัวหรือโทร 1441"
)


class AIClient:
    """เลือกผู้ให้บริการอัตโนมัติ: gemini → claude → ollama → none"""

    def __init__(self):
        self.provider = "none"
        self._gemini = None
        self._init_provider()

    # -- การเลือกผู้ให้บริการ -------------------------------------------------
    def _init_provider(self):
        want = os.environ.get("AI_PROVIDER", "auto").strip().lower()
        order = [want] if want in ("gemini", "claude", "ollama", "none") else ["gemini", "claude", "ollama"]
        for name in order:
            if name == "none":
                break
            try:
                if getattr(self, f"_try_{name}")():
                    self.provider = name
                    return
            except Exception:
                continue
        self.provider = "none"

    def _try_gemini(self) -> bool:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            return False
        from google import genai  # import เมื่อจำเป็นเท่านั้น
        self._gemini = genai.Client(api_key=key)
        self._gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return True

    def _try_claude(self) -> bool:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            return False
        self._claude_key = key
        self._claude_model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        return True

    def _try_ollama(self) -> bool:
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        try:
            r = httpx.get(f"{url}/api/tags", timeout=2)
            r.raise_for_status()
        except Exception:
            return False
        self._ollama_url = url
        self._ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        return True

    # -- การเรียกโมเดล (ต่อผู้ให้บริการ) --------------------------------------
    def _generate(self, system: str, user: str) -> str:
        """เรียกโมเดลตาม provider ปัจจุบัน คืนข้อความล้วน (raise เมื่อพลาด)"""
        if self.provider == "gemini":
            from google.genai import types
            resp = self._gemini.models.generate_content(
                model=self._gemini_model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.4,
                ),
            )
            return (resp.text or "").strip()
        if self.provider == "claude":
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._claude_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._claude_model,
                    "max_tokens": 1024,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"].strip()
        if self.provider == "ollama":
            r = httpx.post(
                f"{self._ollama_url}/api/chat",
                json={
                    "model": self._ollama_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        raise RuntimeError("no provider")

    # -- ฟีเจอร์: วิเคราะห์ข้อความ --------------------------------------------
    def analyze_text(self, text: str) -> dict:
        """วิเคราะห์ข้อความ/SMS คืน dict พร้อม disclaimer เสมอ

        ถ้า AI ใช้ไม่ได้ (ไม่มี key / เรียกพลาด) → ตกลงมาใช้กฎพื้นฐานเสมอ
        """
        matches = scam_data.match_patterns(text)
        result = None
        if self.provider != "none":
            try:
                raw = self._generate(SYSTEM_PROMPT_ANALYZE, f"ข้อความที่ต้องวิเคราะห์:\n{text}")
                result = self._parse_analysis_json(raw)
            except Exception:
                result = None
        if result is None:
            result = self._rule_based_analysis(text, matches)
            result["mode"] = "rules"
        else:
            result["mode"] = "ai"
            # เสริมจุดสังเกตจากกฎพื้นฐานที่ AI อาจไม่ได้พูดถึง
            for m in matches[:2]:
                note = f"เข้าข่าย: {m['name_th']}"
                if note not in result["signals_th"]:
                    result["signals_th"].append(note)
        result["matched_patterns"] = matches
        result["disclaimer_th"] = DISCLAIMER_TEXT_TH
        return result

    @staticmethod
    def _parse_analysis_json(raw: str):
        """ดึง JSON จากคำตอบโมเดล (ตัด markdown fence ถ้ามี)"""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1:
            return None
        data = json.loads(cleaned[start:end + 1])
        risk = max(0, min(95, int(data.get("risk_percent", 0))))
        signals = [str(s) for s in data.get("signals_th", [])][:5]
        return {
            "risk_percent": risk,
            "verdict_th": str(data.get("verdict_th", "")).strip() or "วิเคราะห์แล้ว",
            "signals_th": signals,
            "advice_th": str(data.get("advice_th", "")).strip(),
        }

    @staticmethod
    def _rule_based_analysis(text: str, matches) -> dict:
        """โหมดกฎพื้นฐาน — ไม่ต้องมี key ใด ๆ"""
        risk = scam_data.risk_from_matches(matches)
        if matches:
            top = matches[0]
            signals = []
            for m in matches[:3]:
                words = "“" + "”, “".join(m["matched_words"][:3]) + "”"
                signals.append(f"พบคำที่มิจฉาชีพใช้บ่อย: {words} (เข้าข่าย: {m['name_th']})")
            return {
                "risk_percent": risk,
                "verdict_th": f"ข้อความนี้มีลักษณะคล้าย “{top['name_th']}” ควรระวังอย่างมาก",
                "signals_th": signals,
                "advice_th": top["warning_th"] + " ถ้าเผลอโอนเงินไปแล้ว โทร 1441 ทันที",
            }
        return {
            "risk_percent": risk,
            "verdict_th": "ไม่พบคำที่เข้าข่ายมิจฉาชีพในข้อความนี้",
            "signals_th": ["ไม่พบคำเตือนภัยที่รู้จัก แต่โปรดใช้กฎทอง 3 ข้อประกอบเสมอ"],
            "advice_th": "ถึงจะไม่พบสัญญาณอันตราย ก็อย่าโอนเงินหรือบอกรหัสให้ใครที่ติดต่อมาเอง",
        }

    # -- ฟีเจอร์: แชท -----------------------------------------------------------
    def chat(self, message: str, history: list | None = None) -> dict:
        """แชทกับผู้ช่วย คืน {'reply_th': ..., 'mode': 'ai'|'rules'}"""
        if self.provider != "none":
            try:
                convo = ""
                for turn in (history or [])[-6:]:
                    who = "ผู้ใช้" if turn.get("role") == "user" else "ผู้ช่วย"
                    convo += f"{who}: {turn.get('text', '')}\n"
                convo += f"ผู้ใช้: {message}\nผู้ช่วย:"
                reply = self._generate(SYSTEM_PROMPT_CHAT, convo)
                if reply:
                    return {"reply_th": reply, "mode": "ai"}
            except Exception:
                pass
        return {"reply_th": self._rule_based_chat(message), "mode": "rules"}

    @staticmethod
    def _rule_based_chat(message: str) -> str:
        """แชทโหมดกฎพื้นฐาน — ตอบจากฐานข้อมูลในเครื่อง"""
        msg = message.lower()
        # กำลังถูกหลอก/โอนเงินแล้ว → สายด่วนก่อนเลย
        urgent_words = ["โอนไปแล้ว", "โอนเงินไปแล้ว", "โดนหลอก", "ถูกหลอก", "เสียเงิน", "โดนโกง", "ถูกโกง"]
        if any(w in msg for w in urgent_words):
            return (
                "ใจเย็น ๆ นะคะ ทำตามนี้ทันที:\n"
                "1. โทร 1441 (สายด่วนตำรวจไซเบอร์) เพื่อขออายัดบัญชีปลายทาง ยิ่งเร็วยิ่งมีโอกาสได้เงินคืน\n"
                "2. เก็บหลักฐานทุกอย่าง: สลิปโอนเงิน ข้อความ เบอร์โทร ชื่อบัญชี\n"
                "3. แจ้งความออนไลน์ได้ที่ thaipoliceonline.go.th\n"
                "คุณไม่ได้ผิดนะคะ มิจฉาชีพเก่งเรื่องหลอกมาก ขอให้กำลังใจค่ะ"
            )
        matches = scam_data.match_patterns(message)
        if matches:
            top = matches[0]
            return (
                f"จากที่เล่ามา มีลักษณะคล้าย “{top['name_th']}” ค่ะ\n\n"
                f"{top['warning_th']}\n\n"
                "และจำกฎทอง 3 ข้อไว้เสมอ: เร่งรีบ=อันตราย, วางสายแล้วโทรกลับเบอร์จริงเอง, "
                "ไม่โอนเงิน ไม่บอกรหัส ไม่ว่ากรณีใดค่ะ"
            )
        rule_words = ["กฎทอง", "ป้องกัน", "ทำยังไง", "ทำอย่างไร", "ระวัง"]
        if any(w in msg for w in rule_words):
            lines = ["กฎทอง 3 ข้อ ป้องกันมิจฉาชีพค่ะ:"]
            for r in scam_data.GOLDEN_RULES:
                lines.append(f"{r['icon']} {r['title_th']}")
            lines.append("\nสงสัยอะไรถามเพิ่มได้เลยนะคะ หรือลองไปหน้า “เรียนรู้” เพื่อดูรายละเอียดค่ะ")
            return "\n".join(lines)
        return (
            "สวัสดีค่ะ ฉันคือผู้ช่วยเรื่องภัยออนไลน์นะคะ ถามได้เลย เช่น\n"
            "• “คลิปดาราชวนลงทุน จริงไหม”\n"
            "• “มีคนโทรมาอ้างเป็นตำรวจ ทำยังไงดี”\n"
            "• “โอนเงินไปแล้ว ทำยังไงดี”\n"
            "ถ้าเจอเรื่องด่วน โทร 1441 (ตำรวจไซเบอร์) ได้ตลอด 24 ชั่วโมงค่ะ"
        )


# instance เดียวใช้ทั้งแอป (สร้างตอน import — อ่าน .env แล้วโดย main.py)
_client = None


def get_client() -> AIClient:
    global _client
    if _client is None:
        _client = AIClient()
    return _client
