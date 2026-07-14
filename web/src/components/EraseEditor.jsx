import React, { useEffect, useRef, useState } from "react";
import { useApp } from "../ctx.js";
import { api } from "../api.js";

export default function EraseEditor({ photo, imgRef, onClose }) {
  const app = useApp();
  const [rect, setRect] = useState(null);
  const [brush, setBrush] = useState(30);
  const [method, setMethod] = useState("content-aware");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [hasSaved, setHasSaved] = useState(false);
  const canvasRef = useRef(null);
  const strokes = useRef([]);
  const drawing = useRef(null);

  const measure = () => {
    const img = imgRef.current; if (!img) return;
    const r = img.getBoundingClientRect();
    setRect({ left: r.left, top: r.top, w: r.width, h: r.height });
  };
  useEffect(() => {
    const img = imgRef.current;
    if (img && img.complete && img.naturalWidth) measure();
    else if (img) img.addEventListener("load", measure, { once: true });
    window.addEventListener("resize", measure);
    api.get(`/api/photos/${photo.id}`).then((d) => setHasSaved(!!(d.retouch?.erase?.mask))).catch(() => {});
    return () => window.removeEventListener("resize", measure);
  }, []);

  useEffect(() => { redraw(); }, [rect]);

  const redraw = () => {
    const c = canvasRef.current; if (!c || !rect) return;
    c.width = rect.w; c.height = rect.h;
    paint(c.getContext("2d"), rect.w, rect.h, "rgba(255,64,64,0.55)");
  };
  const paint = (ctx, w, h, color) => {
    ctx.clearRect(0, 0, w, h);
    for (const s of strokes.current) {
      ctx.strokeStyle = ctx.fillStyle = color;
      ctx.lineWidth = s.size * w; ctx.lineCap = ctx.lineJoin = "round";
      ctx.beginPath();
      s.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h)));
      if (s.pts.length === 1) { ctx.arc(s.pts[0][0] * w, s.pts[0][1] * h, ctx.lineWidth / 2, 0, 7); ctx.fill(); }
      else ctx.stroke();
    }
  };
  const pt = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  };
  const down = (e) => {
    e.preventDefault(); canvasRef.current.setPointerCapture(e.pointerId);
    drawing.current = { pts: [pt(e)], size: brush / canvasRef.current.width };
    strokes.current.push(drawing.current); redraw();
  };
  const move = (e) => { if (drawing.current) { drawing.current.pts.push(pt(e)); redraw(); } };
  const up = () => { drawing.current = null; };

  const exportMask = () => {
    const img = imgRef.current;
    const nw = img.naturalWidth || rect.w, nh = img.naturalHeight || rect.h;
    const scale = Math.min(1, 1600 / Math.max(nw, nh));
    const c = document.createElement("canvas");
    c.width = Math.round(nw * scale); c.height = Math.round(nh * scale);
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, c.width, c.height);
    paint(ctx, c.width, c.height, "#fff");
    return c.toDataURL("image/png");
  };
  const apply = async () => {
    if (!strokes.current.length) { alert("Paint over the parts to remove first."); return; }
    setBusy(true);
    try {
      await api.post(`/api/photos/${photo.id}/erase`, { mask: exportMask(), method, prompt: prompt.trim() });
      onClose(); app.setJobsOpen(true); app.refreshJobs();
    } catch (e) { alert("Erase failed: " + e.message); }
    finally { setBusy(false); }
  };
  const dropMask = async () => {
    if (!confirm("Delete this photo's saved erase mask and re-render?")) return;
    await api.post(`/api/photos/${photo.id}/erase`, { clear: true });
    onClose(); app.setJobsOpen(true); app.refreshJobs();
  };

  return (
    <>
      <div className="subbar">
        <span className="muted">Paint over what should disappear</span>
        <label>brush <input type="range" min="6" max="90" value={brush} onChange={(e) => setBrush(+e.target.value)} /></label>
        <button onClick={() => { strokes.current.pop(); redraw(); }}>Undo</button>
        <button onClick={() => { strokes.current = []; redraw(); }}>Clear</button>
        <select value={method} onChange={(e) => setMethod(e.target.value)}>
          <option value="content-aware">Content-Aware Fill (fast)</option>
          <option value="generative">Generative Fill (ComfyUI)</option>
        </select>
        {method === "generative" && (
          <input type="text" placeholder="what to paint instead (optional)" value={prompt}
            onChange={(e) => setPrompt(e.target.value)} style={{ width: 200 }} />
        )}
        {hasSaved && <button onClick={dropMask}>Delete saved mask</button>}
        <span className="spacer" />
        <button onClick={onClose}>Cancel</button>
        <button className="primary" disabled={busy} onClick={apply}>Remove &amp; render</button>
      </div>
      {rect && (
        <canvas ref={canvasRef} className="canvas-overlay"
          style={{ position: "fixed", left: rect.left, top: rect.top, width: rect.w, height: rect.h, zIndex: 65, cursor: "crosshair" }}
          onPointerDown={down} onPointerMove={move} onPointerUp={up} />
      )}
    </>
  );
}
