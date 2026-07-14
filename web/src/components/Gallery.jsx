import React, { useMemo, useRef } from "react";
import { useApp } from "../ctx.js";
import { SHARE, thumbUrl } from "../api.js";
import { visibleFilter } from "./Header.jsx";

export default function Gallery() {
  const app = useApp();
  const lastIdx = useRef(null);
  const vis = useMemo(() => app.photos.filter(visibleFilter(app)), [app.photos, app.filter, app.search, app.albumView, app.decisionMode]);

  const toggleSel = (id, shift) => {
    const idx = vis.findIndex((p) => p.id === id);
    const next = new Set(app.selected);
    if (shift && lastIdx.current != null) {
      const [a, b] = [Math.min(idx, lastIdx.current), Math.max(idx, lastIdx.current)];
      const target = !next.has(id);
      for (let i = a; i <= b; i++) target ? next.add(vis[i].id) : next.delete(vis[i].id);
    } else {
      next.has(id) ? next.delete(id) : next.add(id);
    }
    lastIdx.current = idx;
    app.setSelected(next);
  };

  if (!vis.length) return <main><div className="empty">No photos match.</div></main>;

  return (
    <main>
      {vis.map((p) => (
        <Card key={p.id} p={p} app={app} onToggle={toggleSel} />
      ))}
    </main>
  );
}

function Card({ p, app, onToggle }) {
  const inDecision = app.decisionMode;
  const d = inDecision ? app.decisionOf(p) : null;
  const notsel = !SHARE && !app.albumView && p.selected === 0;
  const qbadge = !SHARE && p.quality_flags && p.quality_flags.length
    ? <span className="qbadge" title={p.quality_reason || ""}>⚠ {p.quality_flags[0]}</span> : null;

  return (
    <div className={"card" + (app.selected.has(p.id) ? " sel" : "") + (notsel ? " notsel" : "")}>
      {!SHARE && (
        <input type="checkbox" className="ck" checked={app.selected.has(p.id)}
          onClick={(e) => { e.stopPropagation(); onToggle(p.id, e.shiftKey); }} readOnly />
      )}
      <div className="imgwrap" onClick={() => app.setLightbox(p.id)}>
        {p.has_output || p.has_preview
          ? <img loading="lazy" src={thumbUrl(p.id, p.output_mtime)} alt="" />
          : <span className="noimg">no image yet</span>}
      </div>
      <div className="meta">
        <span className="id">#{p.id}</span>
        <span className="name" title={p.scene || p.filename}>{p.filename}</span>
        {p.is_portrait && <span title="portrait">👤</span>}
        {qbadge}
        {SHARE ? (
          <span className="decide">
            <button className={"sel" + (d === 1 ? " on" : "")} title="keep"
              onClick={() => app.postDecision(p.id, d === 1 ? "clear" : "select")}>✓</button>
            <button className={"dis" + (d === 0 ? " on" : "")} title="discard"
              onClick={() => app.postDecision(p.id, d === 0 ? "clear" : "discard")}>✗</button>
          </span>
        ) : app.albumView ? (
          <>
            {d === 1 && <span className="badge done">✓</span>}
            {d === 0 && <span className="badge failed">✗</span>}
            <span className={"badge " + p.state}>{p.state}</span>
          </>
        ) : (
          <>
            <span className={"badge " + p.state}>{p.state}</span>
            <span className="decide" title="operator selection">
              <button className={"sel" + (p.selected === 1 ? " on" : "")}
                onClick={() => app.operatorSelect(p.id, p.selected === 1 ? "clear" : "select")}>✓</button>
              <button className={"dis" + (p.selected === 0 ? " on" : "")}
                onClick={() => app.operatorSelect(p.id, p.selected === 0 ? "clear" : "deselect")}>✗</button>
            </span>
          </>
        )}
      </div>
    </div>
  );
}
