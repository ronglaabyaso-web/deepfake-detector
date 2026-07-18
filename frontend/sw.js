/* service worker — ให้ติดตั้งเป็นแอปได้ + เปิดหน้าแอปได้แม้เน็ตช้า
   กลยุทธ์: หน้าเว็บ/ไฟล์แอปใช้ network-first (ได้ของใหม่เสมอ ถ้าเน็ตล่มใช้ของเก็บไว้)
   ส่วน /api ไม่เก็บเลย (ผลตรวจต้องสดเสมอ) */
const CACHE = "fakeclip-helper-v4";
const APP_FILES = ["./", "./index.html", "./manifest.json", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(APP_FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.includes("/api/")) return; // API สดเสมอ
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(e.request).then((hit) => hit || caches.match("./index.html")))
  );
});
