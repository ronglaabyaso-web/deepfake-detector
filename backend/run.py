# -*- coding: utf-8 -*-
"""รันเซิร์ฟเวอร์: python run.py แล้วเปิด http://localhost:8000"""
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print()
    print("  ตัวช่วยจับคลิปปลอม กำลังเริ่มทำงาน...")
    print(f"  เปิดเบราว์เซอร์ไปที่  http://localhost:{port}")
    print(f"  (เครื่องอื่นในวง Wi-Fi เดียวกัน ใช้ IP ของเครื่องนี้แทน localhost)")
    print()
    uvicorn.run("main:app", host=host, port=port)
