import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../ctx.js";
import { api, SHARE, imgUrl } from "../api.js";
import { visibleFilter } from "./Header.jsx";
import CropEditor from "./CropEditor.jsx";
import EraseEditor from "./EraseEditor.jsx";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export default function Lightbox() {
  const app = useApp();
  const vis = useMemo(() => app.photos.filter(visibleFilter(app)), [app.photos, app.filter, app.search, app.albumView]);
  const p = app.photos.find((x) => x.id === app.lightbox);

  const [orig, setOrig] = useState(false);
  const [mode, setMode] = useState("view");     // view | crop | erase
  const [showParams, setShowParams] = useState(false);
  const [detail, setDetail] = useState(null);
  const [zoom, setZoom] = useState({ scale: 1, tx: 0, ty: 0 });
  const wrapRef = useRef(null);
  const imgRef = useRef(null);
  const panning = useRef(null);

  const close = () => app.setLightbox(null);
  const step = useCallback((d) => {
    const i = vis.findIndex((x) => x.id === app.lightbox);
    if (i < 0) return;
    const n = (i + d + vis.length) % vis.length;
    setOrig(false); setMode("view"); setShowParams(false); setZoom({ scale: 1, tx: 0, ty: 0 });
    app.setLightbox(vis[n].id);
  }, [vis, app]);

  useEffect(() => {
    const onKey = (e) => {
      if (mode !== "view") return;
      if (e.key === "Escape") close();
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, mode]);

  const resetZoom = () => setZoom({ scale: 1, tx: 0, ty: 0 });
  const zoomAt = (factor, cx, cy) => {
    if (mode !== "view") return;
    setZoom((z) => {
      const wr = wrapRef.current.getBoundingClientRect();
      const px = (cx ?? wr.left + wr.width / 2) - (wr.left + wr.width / 2);
      const py = (cy ?? wr.top + wr.height / 2) - (wr.top + wr.height / 2);
      const s1 = clamp(z.scale * factor, 1, 6);
      if (s1 === z.scale) return z;
      let tx = px - (s1 / z.scale) * (px - z.tx);
      let ty = py - (s1 / z.scale) * (py - z.ty);
      if (s1 === 1) { tx = 0; ty = 0; }
      return { scale: s1, tx, ty };
    });
  };
  const onWheel = (e) => { if (mode === "view") { e.preventDefault(); zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX, e.clientY); } };
  const onDown = (e) => { if (mode === "view" && zoom.scale > 1) { panning.current = { x: e.clientX, y: e.clientY }; wrapRef.current.setPointerCapture(e.pointerId); } };
  const onMove = (e) => {
    if (!panning.current) return;
    const dx = e.clientX - panning.current.x, dy = e.clientY - panning.current.y;
    panning.current = { x: e.clientX, y: e.clientY };
    setZoom((z) => ({ ...z, tx: z.tx + dx, ty: z.ty + dy }));
  };
  const onUp = () => { panning.current = null; };

  const toggleParams = async () => {
    if (showParams) { setShowParams(false); return; }
    setDetail(await api.get(`/api/photos/${app.lightbox}`));
    setShowParams(true);
  };

  const redoThis = () => { app.setSelected(new Set([app.lightbox])); close(); app.openWizard([app.lightbox]); };

  if (!p) return null;
  const d = SHARE ? p.decision ?? null : null;
  const canDevelop = !SHARE || SHARE.permission === "develop";

  return (
    <div className="lb">
      <div className="bar">
        <b>#{p.id} {p.filename}</b>
        <span className={"badge " + p.state}>{p.state}</span>
        <span className="spacer" />
        {SHARE && (
          <span className="decide">
            <button className={"sel" + (d === 1 ? " on" : "")}
              onClick={() => app.postDecision(p.id, d === 1 ? "clear" : "select")}>✓ Select</button>
            <button className={"dis" + (d === 0 ? " on" : "")}
              onClick={() => app.postDecision(p.id, d === 0 ? "clear" : "discard")}>✗ Discard</button>
          </span>
        )}
        {mode === "view" && (
          <span className="zoomgrp">
            <button onClick={() => zoomAt(1 / 1.3)}>−</button>
            <span>{Math.round(zoom.scale * 100)}%</span>
            <button onClick={() => zoomAt(1.3)}>+</button>
            <button onClick={resetZoom}>Fit</button>
          </span>
        )}
        {p.has_preview && p.has_output && mode === "view" &&
          <button onClick={() => setOrig((v) => !v)}>{orig ? "Latest render" : "Original preview"}</button>}
        {canDevelop && !SHARE && mode === "view" && <>
          <button onClick={() => { resetZoom(); setMode("crop"); }}>Crop</button>
          <button onClick={() => { resetZoom(); setMode("erase"); }}>Erase objects</button>
          <button onClick={toggleParams}>Params</button>
          {p.state === "rejected" && <button onClick={() => app.forceProcess(p.id)}>Process anyway</button>}
          <button className="primary" onClick={redoThis}>Redo this…</button>
        </>}
        <button onClick={() => step(-1)}>←</button>
        <button onClick={() => step(1)}>→</button>
        <button onClick={close}>✕</button>
      </div>

      {mode === "crop" && <CropEditor photo={p} imgRef={imgRef} wrapRef={wrapRef} onClose={() => setMode("view")} />}
      {mode === "erase" && <EraseEditor photo={p} imgRef={imgRef} onClose={() => setMode("view")} />}

      <div className="stage">
        <div ref={wrapRef} className={"lbwrap" + (mode === "view" && zoom.scale > 1 ? " grab" : "")}
          onWheel={onWheel} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
          onDoubleClick={(e) => (zoom.scale > 1 ? resetZoom() : zoomAt(2.5, e.clientX, e.clientY))}>
          <img ref={imgRef} alt="" draggable={false}
            src={imgUrl(p.id, { preview: orig || mode === "crop", v: p.output_mtime })}
            style={{ transform: `translate(${zoom.tx}px,${zoom.ty}px) scale(${zoom.scale})` }} />
        </div>
        {showParams && detail && (
          <aside>
            <b>Scene</b><pre>{detail.scene || "—"}</pre>
            <b>Reproduce</b><div className="cmd">{detail.command || ""}</div>
            <b>Develop</b><pre>{JSON.stringify(detail.develop, null, 1)}</pre>
            <b>Retouch</b><pre>{JSON.stringify(detail.retouch, null, 1)}</pre>
          </aside>
        )}
      </div>
    </div>
  );
}
