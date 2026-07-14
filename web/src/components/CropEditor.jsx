import React, { useEffect, useRef, useState } from "react";
import { useApp } from "../ctx.js";
import { api } from "../api.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const RATIOS = [["free", "free"], ["1", "1:1"], ["0.8", "4:5"], ["0.714", "5:7"],
  ["1.5", "3:2"], ["0.667", "2:3"], ["1.778", "16:9"]];

// Renders the preview onto a canvas rotated with an expanding bounding box (gray-filled
// corners) — pixel-for-pixel what the develop stage does (PIL rotate expand=True, then
// crop). The crop box is stored as fractions of that rotated frame, so straighten + 90°
// orientation + crop all map straight onto develop {rotation_deg, crop}.
export default function CropEditor({ photo, imgRef, wrapRef, onClose }) {
  const app = useApp();
  const canvasRef = useRef(null);
  const [orient, setOrient] = useState(0);      // 0 / ±90 / 180  (orientation fix)
  const [straighten, setStraighten] = useState(0); // -45..45 (fine)
  const [ratio, setRatio] = useState("free");
  // default to the full frame so a pure orientation fix (rotate 90° → Apply) doesn't crop;
  // drag a handle inward to actually crop / remove straighten's gray corners
  const [box, setBox] = useState({ x: 0, y: 0, w: 1, h: 1 }); // fractions of canvas
  const [dims, setDims] = useState(null);        // {left,top,cw,ch} of the canvas (screen px)
  const [busy, setBusy] = useState(false);
  const drag = useRef(null);
  const angle = orient + straighten;

  const render = () => {
    const img = imgRef.current, wrap = wrapRef.current, canvas = canvasRef.current;
    if (!img || !wrap || !canvas || !img.naturalWidth) return;
    const wr = wrap.getBoundingClientRect();
    const availW = wr.width - 24, availH = wr.height - 24;
    const sw = img.naturalWidth, sh = img.naturalHeight;
    const rad = (angle * Math.PI) / 180;
    const bbW = Math.abs(sw * Math.cos(rad)) + Math.abs(sh * Math.sin(rad));
    const bbH = Math.abs(sw * Math.sin(rad)) + Math.abs(sh * Math.cos(rad));
    const fit = Math.min(availW / bbW, availH / bbH);
    const cw = Math.max(1, Math.round(bbW * fit)), ch = Math.max(1, Math.round(bbH * fit));
    canvas.width = cw; canvas.height = ch;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#808080"; ctx.fillRect(0, 0, cw, ch);
    ctx.save();
    ctx.translate(cw / 2, ch / 2);
    ctx.rotate(rad);
    ctx.drawImage(img, (-sw * fit) / 2, (-sh * fit) / 2, sw * fit, sh * fit);
    ctx.restore();
    setDims({ left: wr.left + (wr.width - cw) / 2, top: wr.top + (wr.height - ch) / 2, cw, ch });
  };

  useEffect(() => {
    const img = imgRef.current;
    if (img && img.complete && img.naturalWidth) render();
    else if (img) img.addEventListener("load", render, { once: true });
    window.addEventListener("resize", render);
    return () => window.removeEventListener("resize", render);
  }, []);
  useEffect(() => { render(); }, [angle]);

  const fitRatio = (r) => {
    if (r === "free") { setBox({ x: 0, y: 0, w: 1, h: 1 }); return; }
    if (!dims) return;
    const rr = +r, cw = dims.cw, ch = dims.ch;
    let w = cw * 0.9, h = w / rr;
    if (h > ch * 0.9) { h = ch * 0.9; w = h * rr; }
    setBox({ x: (cw - w) / 2 / cw, y: (ch - h) / 2 / ch, w: w / cw, h: h / ch });
  };
  const changeRatio = (r) => { setRatio(r); fitRatio(r); };

  const onDown = (e, mode) => {
    e.preventDefault(); e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    drag.current = { mode, sx: e.clientX, sy: e.clientY, box0: { ...box } };
  };
  const onMove = (e) => {
    if (!drag.current || !dims) return;
    const { mode, sx, sy, box0 } = drag.current;
    const dx = (e.clientX - sx) / dims.cw, dy = (e.clientY - sy) / dims.ch;
    let b;
    if (mode === "move") {
      b = { x: clamp(box0.x + dx, 0, 1 - box0.w), y: clamp(box0.y + dy, 0, 1 - box0.h), w: box0.w, h: box0.h };
    } else {
      let x1 = box0.x, y1 = box0.y, x2 = box0.x + box0.w, y2 = box0.y + box0.h;
      if (mode.includes("w")) x1 = clamp(box0.x + dx, 0, x2 - 0.03);
      if (mode.includes("e")) x2 = clamp(box0.x + box0.w + dx, x1 + 0.03, 1);
      if (mode.includes("n")) y1 = clamp(box0.y + dy, 0, y2 - 0.03);
      if (mode.includes("s")) y2 = clamp(box0.y + box0.h + dy, y1 + 0.03, 1);
      b = { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
      if (ratio !== "free") b = lockRatio(b, mode, +ratio, dims.cw / dims.ch);
    }
    setBox(b);
  };
  const onUp = () => { drag.current = null; };

  const apply = async () => {
    setBusy(true);
    try {
      const crop = { x: box.x, y: box.y, w: box.w, h: box.h };
      const develop = { crop };
      if (Math.abs(angle) > 0.01) develop.rotation_deg = angle;
      await api.post("/api/redo", { ids: [photo.id], overrides: { develop }, from_raw: true });
      onClose(); app.setJobsOpen(true); app.refreshJobs();
    } catch (e) { alert("Crop failed: " + e.message); }
    finally { setBusy(false); }
  };

  const px = dims && { left: dims.left + box.x * dims.cw, top: dims.top + box.y * dims.ch, width: box.w * dims.cw, height: box.h * dims.ch };

  return (
    <>
      <div className="subbar">
        <span className="muted">Straighten &amp; crop; rotate to fix orientation. Re-develops the RAW.</span>
        <button onClick={() => setOrient((o) => o - 90)} title="rotate left 90°">⟲ 90°</button>
        <button onClick={() => setOrient((o) => o + 90)} title="rotate right 90°">⟳ 90°</button>
        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          straighten
          <input type="range" min="-45" max="45" step="0.5" value={straighten}
            onChange={(e) => setStraighten(+e.target.value)} style={{ width: 120 }} />
          <span style={{ minWidth: 34, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{straighten}°</span>
        </label>
        <label>ratio&nbsp;
          <select value={ratio} onChange={(e) => changeRatio(e.target.value)}>
            {RATIOS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </label>
        <span className="spacer" />
        <button onClick={() => { setStraighten(0); setOrient(0); changeRatio("free"); }}>Reset</button>
        <button onClick={onClose}>Cancel</button>
        <button className="primary" disabled={busy} onClick={apply}>Apply &amp; render</button>
      </div>
      <canvas ref={canvasRef} style={{ position: "fixed", zIndex: 64,
        left: dims ? dims.left : -99999, top: dims ? dims.top : -99999 }} />
      {px && (
        <div className="canvas-overlay" style={{ position: "fixed", left: dims.left, top: dims.top, width: dims.cw, height: dims.ch, zIndex: 65 }}
          onPointerMove={onMove} onPointerUp={onUp}>
          <div id="cropbox" style={{ left: px.left - dims.left, top: px.top - dims.top, width: px.width, height: px.height }}
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

// keep the aspect ratio while dragging a corner; ratio is width/height in *canvas px*,
// but box is fractions, so convert using the canvas aspect (caspect = cw/ch)
function lockRatio(b, mode, ratioWH, caspect) {
  // work in a square-normalized space so ratio math is correct despite fraction axes
  const rx = ratioWH / caspect; // target w/h in fraction units
  const anchorX = mode.includes("w") ? b.x + b.w : b.x;
  const anchorY = mode.includes("n") ? b.y + b.h : b.y;
  let w = b.w, h = w / rx;
  h = Math.min(h, mode.includes("n") ? anchorY : 1 - anchorY); w = h * rx;
  w = Math.min(w, mode.includes("w") ? anchorX : 1 - anchorX); h = w / rx;
  return { x: mode.includes("w") ? anchorX - w : anchorX, y: mode.includes("n") ? anchorY - h : anchorY, w, h };
}
