import React, { useEffect, useRef, useState } from "react";
import { useApp } from "../ctx.js";
import { api } from "../api.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const RATIOS = [["free", "free"], ["1", "1:1"], ["0.8", "4:5"], ["0.714", "5:7"],
  ["1.5", "3:2"], ["0.667", "2:3"], ["1.778", "16:9"]];

export default function CropEditor({ photo, imgRef, onClose }) {
  const app = useApp();
  const [rect, setRect] = useState(null);        // image bounding rect (fixed coords)
  const [ratio, setRatio] = useState("free");
  const [box, setBox] = useState(null);          // {x,y,w,h} in image px
  const [busy, setBusy] = useState(false);
  const drag = useRef(null);

  const measure = () => {
    const img = imgRef.current;
    if (!img) return;
    const r = img.getBoundingClientRect();
    setRect({ left: r.left, top: r.top, w: r.width, h: r.height });
    setBox((b) => b || initBox(r.width, r.height, "free"));
  };
  useEffect(() => {
    // image may still be loading (src switched to preview)
    const img = imgRef.current;
    if (img && img.complete && img.naturalWidth) measure();
    else if (img) img.addEventListener("load", measure, { once: true });
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  function initBox(w, h, r) {
    let cw = w * 0.8, ch = h * 0.8;
    if (r !== "free") { const rr = +r; cw = w * 0.9; ch = cw / rr; if (ch > h * 0.9) { ch = h * 0.9; cw = ch * rr; } }
    return { x: (w - cw) / 2, y: (h - ch) / 2, w: cw, h: ch };
  }
  const changeRatio = (r) => { setRatio(r); if (rect) setBox(initBox(rect.w, rect.h, r)); };

  const onDown = (e, mode) => {
    e.preventDefault(); e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    drag.current = { mode, sx: e.clientX, sy: e.clientY, box0: { ...box } };
  };
  const onMove = (e) => {
    if (!drag.current || !rect) return;
    const { mode, sx, sy, box0 } = drag.current;
    const dx = e.clientX - sx, dy = e.clientY - sy, W = rect.w, H = rect.h;
    let b;
    if (mode === "move") {
      b = { x: clamp(box0.x + dx, 0, W - box0.w), y: clamp(box0.y + dy, 0, H - box0.h), w: box0.w, h: box0.h };
    } else {
      let x1 = box0.x, y1 = box0.y, x2 = box0.x + box0.w, y2 = box0.y + box0.h;
      if (mode.includes("w")) x1 = clamp(box0.x + dx, 0, x2 - 20);
      if (mode.includes("e")) x2 = clamp(box0.x + box0.w + dx, x1 + 20, W);
      if (mode.includes("n")) y1 = clamp(box0.y + dy, 0, y2 - 20);
      if (mode.includes("s")) y2 = clamp(box0.y + box0.h + dy, y1 + 20, H);
      b = { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
      if (ratio !== "free") b = lockRatio(b, mode, +ratio, W, H);
    }
    setBox(b);
  };
  const onUp = () => { drag.current = null; };

  const apply = async () => {
    if (!box || !rect) return;
    setBusy(true);
    try {
      const crop = { x: box.x / rect.w, y: box.y / rect.h, w: box.w / rect.w, h: box.h / rect.h };
      await api.post("/api/redo", { ids: [photo.id], overrides: { develop: { crop } }, from_raw: true });
      onClose(); app.setJobsOpen(true); app.refreshJobs();
    } catch (e) { alert("Crop failed: " + e.message); }
    finally { setBusy(false); }
  };

  return (
    <>
      <div className="subbar">
        <span className="muted">Drag the box; corners resize. Cropping re-develops the RAW.</span>
        <label>ratio&nbsp;
          <select value={ratio} onChange={(e) => changeRatio(e.target.value)}>
            {RATIOS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </label>
        <span className="spacer" />
        <button onClick={() => rect && setBox(initBox(rect.w, rect.h, ratio))}>Reset</button>
        <button onClick={onClose}>Cancel</button>
        <button className="primary" disabled={busy} onClick={apply}>Apply crop &amp; render</button>
      </div>
      {rect && box && (
        <div className="canvas-overlay" style={{ position: "fixed", left: rect.left, top: rect.top, width: rect.w, height: rect.h, zIndex: 65 }}
          onPointerMove={onMove} onPointerUp={onUp}>
          <div id="cropbox" style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
            onPointerDown={(e) => onDown(e, "move")}>
            {["nw", "ne", "sw", "se"].map((c) => (
              <span key={c} className={"chandle " + c} onPointerDown={(e) => onDown(e, c)} />
            ))}
          </div>
        </div>
      )}
    </>
  );
}

function lockRatio(b, mode, ratio, W, H) {
  const anchorX = mode.includes("w") ? b.x + b.w : b.x;
  const anchorY = mode.includes("n") ? b.y + b.h : b.y;
  let w = b.w, h = w / ratio;
  h = Math.min(h, mode.includes("n") ? anchorY : H - anchorY); w = h * ratio;
  w = Math.min(w, mode.includes("w") ? anchorX : W - anchorX); h = w / ratio;
  return { x: mode.includes("w") ? anchorX - w : anchorX, y: mode.includes("n") ? anchorY - h : anchorY, w, h };
}
