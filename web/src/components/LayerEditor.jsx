import React, { useEffect, useRef, useState } from "react";
import { useApp } from "../ctx.js";
import { api, API } from "../api.js";

const LAYERS = [
  ["backdrop", "Backdrop clean layer"],
  ["floor", "Floor reflection layer"],
];
const MODES = [
  ["add", "Add", "rgba(60,220,120,0.55)", 255],
  ["remove", "Remove", "rgba(255,70,70,0.55)", 0],
  ["auto", "Auto", null, 128],
];

// Correct the background layers the pipeline made automatically. The automatic layer (as
// saved by the photo's last render) shows in blue; paint Add (force the layer there),
// Remove (keep the photo as shot there) or Auto (drop your correction). The edit is saved
// as a gray mask (128 = auto) and applied on every render of this photo.
export default function LayerEditor({ photo, imgRef, onClose }) {
  const app = useApp();
  const [rect, setRect] = useState(null);
  const [info, setInfo] = useState(null);
  const [layer, setLayer] = useState("backdrop");
  const [mode, setMode] = useState("remove");
  const [brush, setBrush] = useState(40);
  const [showAuto, setShowAuto] = useState(true);
  const [size, setSize] = useState(null);          // mask resolution {w,h}
  const [busy, setBusy] = useState(false);
  const autoRef = useRef(null);    // tinted automatic layer (display)
  const paintRef = useRef(null);   // coloured edit strokes (display)
  const maskRef = useRef(null);    // the edit mask itself (offscreen, gray)
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
    api.get(`/api/photos/${photo.id}/layers`).then((d) => setInfo(d.layers)).catch(() => setInfo({}));
    return () => window.removeEventListener("resize", measure);
  }, []);

  const loadImage = (src) => new Promise((res) => {
    const im = new Image();
    im.onload = () => res(im); im.onerror = () => res(null); im.src = src;
  });

  // (re)load the automatic layer + any saved edit whenever the layer changes
  useEffect(() => {
    if (!info) return;
    let cancelled = false;
    (async () => {
      const li = info[layer] || {};
      const base = `${API}/api/photos/${photo.id}/layer/${layer}`;
      const auto = li.auto ? await loadImage(`${base}?t=${li.mtime}`) : null;
      const edit = li.edit ? await loadImage(`${base}?edit=1&t=${Date.now()}`) : null;
      if (cancelled) return;
      const nat = imgRef.current;
      const s = Math.min(1, 1600 / Math.max(nat.naturalWidth || 1600, nat.naturalHeight || 1600));
      const w = auto ? auto.naturalWidth : Math.round((nat.naturalWidth || 1600) * s);
      const h = auto ? auto.naturalHeight : Math.round((nat.naturalHeight || 1600) * s);
      setSize({ w, h });
      // offscreen edit mask, gray = auto
      const mk = document.createElement("canvas"); mk.width = w; mk.height = h;
      const mctx = mk.getContext("2d");
      mctx.fillStyle = "rgb(128,128,128)"; mctx.fillRect(0, 0, w, h);
      if (edit) mctx.drawImage(edit, 0, 0, w, h);
      maskRef.current = mk;
      requestAnimationFrame(() => {
        drawAuto(auto, w, h);
        rebuildPaint(w, h);
      });
    })();
    return () => { cancelled = true; };
  }, [info, layer]);

  const drawAuto = (auto, w, h) => {
    const c = autoRef.current; if (!c) return;
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (!auto) return;
    ctx.drawImage(auto, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h);
    for (let i = 0; i < d.data.length; i += 4) {
      const v = d.data[i];
      d.data[i] = 70; d.data[i + 1] = 140; d.data[i + 2] = 255; d.data[i + 3] = v * 0.5;
    }
    ctx.putImageData(d, 0, 0);
  };
  // colour the saved edit: >128 add (green), <128 remove (red)
  const rebuildPaint = (w, h) => {
    const c = paintRef.current, mk = maskRef.current; if (!c || !mk) return;
    c.width = w; c.height = h;
    const src = mk.getContext("2d").getImageData(0, 0, w, h);
    const ctx = c.getContext("2d");
    const out = ctx.createImageData(w, h);
    for (let i = 0; i < src.data.length; i += 4) {
      const v = src.data[i] - 128;
      if (v > 20) { out.data[i] = 60; out.data[i + 1] = 220; out.data[i + 2] = 120; out.data[i + 3] = Math.min(140, v * 1.1); }
      else if (v < -20) { out.data[i] = 255; out.data[i + 1] = 70; out.data[i + 2] = 70; out.data[i + 3] = Math.min(140, -v * 1.1); }
    }
    ctx.putImageData(out, 0, 0);
  };

  const pt = (e) => {
    const r = paintRef.current.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * size.w, ((e.clientY - r.top) / r.height) * size.h];
  };
  const stroke = (from, to) => {
    const [, , color, gray] = MODES.find(([k]) => k === mode);
    const lw = (brush / rect.w) * size.w;
    for (const [ctx, style, erase] of [
      [maskRef.current.getContext("2d"), `rgb(${gray},${gray},${gray})`, false],
      [paintRef.current.getContext("2d"), color, mode === "auto"],
    ]) {
      ctx.save();
      ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
      ctx.strokeStyle = ctx.fillStyle = erase ? "#000" : style;
      ctx.lineWidth = lw; ctx.lineCap = ctx.lineJoin = "round";
      ctx.beginPath(); ctx.moveTo(...from); ctx.lineTo(...to); ctx.stroke();
      ctx.restore();
    }
  };
  const down = (e) => {
    if (!size) return;
    e.preventDefault(); paintRef.current.setPointerCapture(e.pointerId);
    const p = pt(e); drawing.current = p; stroke(p, p);
  };
  const move = (e) => { if (drawing.current) { const p = pt(e); stroke(drawing.current, p); drawing.current = p; } };
  const up = () => {
    if (drawing.current) rebuildPaint(size.w, size.h);   // normalise the colouring
    drawing.current = null;
  };

  const clearStrokes = () => {
    const mk = maskRef.current; if (!mk) return;
    const ctx = mk.getContext("2d");
    ctx.fillStyle = "rgb(128,128,128)"; ctx.fillRect(0, 0, size.w, size.h);
    rebuildPaint(size.w, size.h);
  };
  const save = async (clear) => {
    setBusy(true);
    try {
      await api.post(`/api/photos/${photo.id}/layer/${layer}`,
        clear ? { clear: true } : { mask: maskRef.current.toDataURL("image/png") });
      onClose(); app.setJobsOpen(true); app.refreshJobs();
    } catch (e) { alert("Saving the layer failed: " + e.message); }
    finally { setBusy(false); }
  };

  const li = info?.[layer] || {};
  return (
    <>
      <div className="subbar">
        <select value={layer} onChange={(e) => setLayer(e.target.value)} title="which automatic layer to correct">
          {LAYERS.map(([k, l]) => <option key={k} value={k}>{l}{info?.[k]?.edit ? " (edited)" : ""}</option>)}
        </select>
        <span className="seggrp">
          {MODES.map(([k, l]) => (
            <button key={k} className={"seg" + (mode === k ? " on" : "")} onClick={() => setMode(k)}
              title={k === "add" ? "force the layer here" : k === "remove" ? "keep the photo as shot here" : "undo your correction here"}>{l}</button>
          ))}
        </span>
        <label>brush <input type="range" min="6" max="160" value={brush} onChange={(e) => setBrush(+e.target.value)} /></label>
        <label><input type="checkbox" checked={showAuto} onChange={(e) => setShowAuto(e.target.checked)} /> show automatic layer</label>
        <button onClick={clearStrokes} title="set everything back to automatic (not saved yet)">Clear edits</button>
        {li.edit && <button onClick={() => save(true)} disabled={busy} title="delete the saved correction and re-render">Delete saved edit</button>}
        {info && !li.auto && <span className="muted">No automatic layer yet — render with clean/expand{layer === "floor" ? " + a floor look" : ""} first.</span>}
        <span className="spacer" />
        <button onClick={onClose}>Cancel</button>
        <button className="primary" disabled={busy || !size} onClick={() => save(false)}>{busy ? "Saving…" : "Save & render"}</button>
      </div>
      {rect && (
        <div style={{ position: "fixed", left: rect.left, top: rect.top, width: rect.w, height: rect.h, zIndex: 65 }}>
          <canvas ref={autoRef} className="canvas-overlay" style={{ width: "100%", height: "100%", pointerEvents: "none", opacity: showAuto ? 1 : 0 }} />
          <canvas ref={paintRef} className="canvas-overlay" style={{ width: "100%", height: "100%", cursor: "crosshair" }}
            onPointerDown={down} onPointerMove={move} onPointerUp={up} />
        </div>
      )}
    </>
  );
}
