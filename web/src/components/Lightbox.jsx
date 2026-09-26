import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../ctx.js";
import { api, SHARE, imgUrl, thumbUrl } from "../api.js";
import { visibleFilter, sortComparator } from "./Header.jsx";
import CropEditor from "./CropEditor.jsx";
import EraseEditor from "./EraseEditor.jsx";
import AdjustPanel from "./AdjustPanel.jsx";
import LayerEditor from "./LayerEditor.jsx";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export default function Lightbox() {
  const app = useApp();
  const vis = useMemo(() => app.photos.filter(visibleFilter(app)).sort(sortComparator(app)),
    [app.photos, app.filter, app.search, app.albumView, app.sort]);
  const p = app.photos.find((x) => x.id === app.lightbox);

  const [viewOriginal, setViewOriginal] = useState(false); // show the untouched full-res original
  const [mode, setMode] = useState("view");     // view | crop | erase | layers
  const [panel, setPanel] = useState(() => {       // Adjust panel open (remembered)
    try { return localStorage.getItem("pr_adjust_panel") !== "0" && window.innerWidth > 720; } catch { return true; }
  });
  const [zoom, setZoom] = useState({ scale: 1, tx: 0, ty: 0 });
  const [replacing, setReplacing] = useState(false);
  const wrapRef = useRef(null);
  const imgRef = useRef(null);
  const fileRef = useRef(null);
  const panning = useRef(null);
  const swipe = useRef(null);   // horizontal swipe-to-navigate start (only at fit/scale 1)

  const close = () => app.setLightbox(null);
  const step = useCallback((d) => {
    const i = vis.findIndex((x) => x.id === app.lightbox);
    if (i < 0) return;
    const n = (i + d + vis.length) % vis.length;
    setMode("view"); setZoom({ scale: 1, tx: 0, ty: 0 });
    app.setLightbox(vis[n].id);
  }, [vis, app]);

  // reflect each photo's delivery choice: show the original when it's the chosen version
  useEffect(() => { setViewOriginal((p?.deliver || "developed") === "original"); }, [p?.id, p?.deliver]);

  // Select / discard the open photo. Customers (share) set the album decision; operators
  // set their own pick. `advance` moves to the next photo — the fast Z/X culling flow.
  const decide = useCallback((kind, advance) => {
    if (!p) return;
    if (SHARE) app.postDecision(p.id, kind === "select" ? "select" : "discard");
    else app.operatorSelect(p.id, kind === "select" ? "select" : "deselect");
    if (advance) step(1);
  }, [p, app, step]);

  useEffect(() => {
    const onKey = (e) => {
      if (mode !== "view") return;
      const t = e.target;
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;  // don't hijack typing
      if (e.key === "Escape") close();
      else if (e.key === "ArrowLeft") step(-1);
      else if (e.key === "ArrowRight") step(1);
      else if (e.key === "z" || e.key === "Z") { e.preventDefault(); decide("select", true); }
      else if (e.key === "x" || e.key === "X") { e.preventDefault(); decide("discard", true); }
      else if (e.key === "\\") { e.preventDefault(); setViewOriginal((v) => !v); }   // before/after
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, mode, decide]);

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
  const onDown = (e) => {
    if (mode !== "view") return;
    if (zoom.scale > 1) { panning.current = { x: e.clientX, y: e.clientY }; wrapRef.current.setPointerCapture(e.pointerId); }
    else if (vis.length > 1) { swipe.current = { x: e.clientX, y: e.clientY }; }  // swipe-to-navigate
  };
  const onMove = (e) => {
    if (!panning.current) return;
    const dx = e.clientX - panning.current.x, dy = e.clientY - panning.current.y;
    panning.current = { x: e.clientX, y: e.clientY };
    setZoom((z) => ({ ...z, tx: z.tx + dx, ty: z.ty + dy }));
  };
  const onUp = (e) => {
    if (swipe.current) {
      const dx = e.clientX - swipe.current.x, dy = e.clientY - swipe.current.y;
      swipe.current = null;
      if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) step(dx < 0 ? 1 : -1);  // L→R = prev
    }
    panning.current = null;
  };

  const togglePanel = () => setPanel((v) => {
    try { localStorage.setItem("pr_adjust_panel", v ? "0" : "1"); } catch {}
    return !v;
  });

  // keep the current photo visible in the filmstrip
  const stripRef = useRef(null);
  useEffect(() => {
    stripRef.current?.querySelector(".cur")?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [app.lightbox]);

  // Edited vs original: operators set the delivered version (persisted); customers just
  // switch which one they're viewing.
  const chooseVersion = (which) => {
    if (SHARE) setViewOriginal(which === "original");
    else app.setDeliver(p.id, which);
  };

  const onReplaceFile = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";                 // allow re-picking the same file later
    if (!file) return;
    setReplacing(true);
    try { await app.replacePhoto(p.id, file); }
    catch (err) { alert("Replace failed: " + err.message); }
    finally { setReplacing(false); }
  };

  const deleteThis = async () => {
    if (!confirm(`Delete #${p.id} ${p.filename}?\nThis permanently removes the photo and all its files.`)) return;
    const i = vis.findIndex((x) => x.id === p.id);
    const nxt = vis.length > 1 ? vis[(i + 1) % vis.length] : null;
    try { await app.deletePhoto(p.id); }
    catch (e) { return alert("Delete failed: " + e.message); }
    if (nxt && nxt.id !== p.id) app.setLightbox(nxt.id); else close();
  };

  if (!p) return null;
  const d = SHARE ? p.decision ?? null : null;
  // develop needs the RAW; finalized albums have dropped it, so hide those actions
  const canDevelop = (!SHARE || SHARE.permission === "develop") && !p.raw_removed;
  const rendering = app.jobs.some((j) => ["queued", "running"].includes(j.items[String(p.id)]?.status));
  const hasImg = p.has_output || p.has_preview;
  const Tool = ({ icon, label, on, onClick, title, cls = "", disabled }) => (
    <button className={"tool " + cls + (on ? " on" : "")} onClick={onClick} title={title || label} disabled={disabled}>
      <span className="ti">{icon}</span><span className="tl">{label}</span>
    </button>
  );

  return (
    <div className="lb editor">
      <div className="bar">
        <button className="ghost" onClick={close} title="back to the gallery (Esc)">‹ Gallery</button>
        <b className="fname">#{p.id} {p.filename}</b>
        <span className={"badge " + p.state}>{p.state}</span>
        {rendering && <span className="badge analyzed"><span className="live" /> rendering</span>}
        {p.raw_removed && <span className="muted" style={{ fontSize: 12 }} title="album finalized — RAW deleted">finalized · RAW removed</span>}
        <span className="spacer" />
        {SHARE && (
          <span className="decide" title="keep (Z) / discard (X)">
            <button className={"sel" + (d === 1 ? " on" : "")}
              onClick={() => app.postDecision(p.id, d === 1 ? "clear" : "select")}>✓ Select</button>
            <button className={"dis" + (d === 0 ? " on" : "")}
              onClick={() => app.postDecision(p.id, d === 0 ? "clear" : "discard")}>✗ Discard</button>
          </span>
        )}
        {!SHARE && mode === "view" && (
          <span className="decide" title="your pick — keep (Z) / cut (X)">
            <button className={"sel" + (p.selected === 1 ? " on" : "")}
              onClick={() => app.operatorSelect(p.id, p.selected === 1 ? "clear" : "select")}>✓ Keep</button>
            <button className={"dis" + (p.selected === 0 ? " on" : "")}
              onClick={() => app.operatorSelect(p.id, p.selected === 0 ? "clear" : "deselect")}>✗ Cut</button>
          </span>
        )}
        {mode === "view" && hasImg && (
          <span className="deliver" title={SHARE ? "view edited or original (\\)" : "which version is delivered / downloaded (\\ toggles the view)"}>
            <button className={"seg" + (!viewOriginal ? " on" : "")}
              onClick={() => chooseVersion("developed")}>Edited</button>
            <button className={"seg" + (viewOriginal ? " on" : "")}
              onClick={() => chooseVersion("original")}>Original</button>
          </span>
        )}
        {mode === "view" && hasImg && (
          <span className="qualgrp" title="download size / quality">
            <select value={app.downloadPreset} onChange={(e) => app.changeDownloadPreset(e.target.value)}>
              <option value="high">High quality</option>
              <option value="print">Print (A4)</option>
              <option value="medium">Medium</option>
              <option value="small">Small</option>
            </select>
          </span>
        )}
        {mode === "view" && hasImg && <button onClick={() => app.downloadPhotos([p.id])}>⬇ Download</button>}
        {canDevelop && mode === "view" &&
          <button className={panel ? "on-accent" : ""} onClick={togglePanel} title="show / hide the Adjust panel">⚙ Adjust</button>}
        <button className="ghost" onClick={close} title="close (Esc)">✕</button>
      </div>

      {mode === "crop" && <CropEditor photo={p} onClose={() => setMode("view")} />}
      {mode === "erase" && <EraseEditor photo={p} imgRef={imgRef} onClose={() => setMode("view")} />}
      {mode === "layers" && <LayerEditor photo={p} imgRef={imgRef} onClose={() => setMode("view")} />}

      <div className="stage">
        {mode === "view" && (
          <nav className="rail">
            <Tool icon="✋" label="View" on={mode === "view"} onClick={() => setMode("view")} title="pan & zoom (wheel / double-click)" />
            {canDevelop && <>
              <Tool icon="⬒" label="Crop" onClick={() => { resetZoom(); setMode("crop"); }} title="crop, rotate & straighten" />
              <Tool icon="⌫" label="Erase" onClick={() => { resetZoom(); setMode("erase"); }} title="paint objects to remove" />
              {!SHARE && <Tool icon="◧" label="Layers" onClick={() => { resetZoom(); setViewOriginal(false); setMode("layers"); }}
                title="correct the background layers made automatically (backdrop clean, floor)" />}
            </>}
            {hasImg && <Tool icon="◑" label="Before" on={viewOriginal} onClick={() => setViewOriginal((v) => !v)}
              title="before / after comparison (\\ key)" />}
            <span className="rail-sep" />
            <Tool icon="−" label="Zoom out" onClick={() => zoomAt(1 / 1.3)} />
            <Tool icon="+" label="Zoom in" onClick={() => zoomAt(1.3)} />
            <Tool icon="⤢" label={`${Math.round(zoom.scale * 100)}%`} onClick={resetZoom} title="fit to screen" />
            <span className="spacer" />
            {!SHARE && p.state === "rejected" && canDevelop &&
              <Tool icon="▶" label="Process" onClick={() => app.forceProcess(p.id)} title="process anyway (quality gate rejected it)" />}
            {!SHARE && <>
              <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={onReplaceFile} />
              <Tool icon="⇪" label={replacing ? "…" : "Replace"} disabled={replacing} onClick={() => fileRef.current?.click()}
                title="replace the edited image with an uploaded file (keeps the name)" />
              <Tool icon="🗑" label="Delete" cls="danger" onClick={deleteThis} title="delete this photo" />
            </>}
          </nav>
        )}

        <div ref={wrapRef} className={"lbwrap" + (mode === "view" && zoom.scale > 1 ? " grab" : "")}
          onWheel={onWheel} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
          onDoubleClick={(e) => (zoom.scale > 1 ? resetZoom() : zoomAt(2.5, e.clientX, e.clientY))}>
          <img ref={imgRef} alt="" draggable={false}
            src={viewOriginal ? imgUrl(p.id, { original: true }) : imgUrl(p.id, { v: p.output_mtime })}
            style={{ transform: `translate(${zoom.tx}px,${zoom.ty}px) scale(${zoom.scale})` }} />
          {viewOriginal && mode === "view" && <span className="stage-tag">Original</span>}
          {vis.length > 1 && mode === "view" && zoom.scale === 1 && <>
            <button className="lb-nav prev" aria-label="Previous photo"
              onPointerDown={(e) => e.stopPropagation()} onClick={() => step(-1)}>‹</button>
            <button className="lb-nav next" aria-label="Next photo"
              onPointerDown={(e) => e.stopPropagation()} onClick={() => step(1)}>›</button>
          </>}
        </div>

        {canDevelop && panel && mode === "view" && <AdjustPanel photo={p} rendering={rendering} />}
      </div>

      {mode === "view" && vis.length > 1 && (
        <div className="filmstrip" ref={stripRef}>
          {vis.map((x) => (
            <button key={x.id} className={"frame" + (x.id === p.id ? " cur" : "")} title={`#${x.id} ${x.filename}`}
              onClick={() => { setMode("view"); resetZoom(); app.setLightbox(x.id); }}>
              {x.has_output || x.has_preview ? <img loading="lazy" src={thumbUrl(x.id, x.output_mtime)} alt="" /> : <span>#{x.id}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
