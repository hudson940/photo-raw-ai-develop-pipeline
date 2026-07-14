import React, { useMemo, useState } from "react";
import { useApp } from "../ctx.js";
import { api, SHARE } from "../api.js";

const STATE_ORDER = ["all", "done", "retouched", "developed", "analyzed", "previewed", "pending", "rejected", "review", "failed"];

export default function Header() {
  const app = useApp();
  const { photos, filter, setFilter, search, setSearch, selected, setSelected,
    decisionMode, decisionOf, albumView, viewAlbum, albums, me, authEnabled, isSuperAdmin } = app;
  const [menuOpen, setMenuOpen] = useState(false);

  const chips = useMemo(() => {
    if (decisionMode) {
      const pool = photos.filter((p) => !albumView || albumView.ids.has(p.id));
      const sel = pool.filter((p) => decisionOf(p) === 1).length;
      const dis = pool.filter((p) => decisionOf(p) === 0).length;
      return [["all", `all ${pool.length}`], ["sel", `✓ selected ${sel}`],
        ["dis", `✗ discarded ${dis}`], ["und", `undecided ${pool.length - sel - dis}`]];
    }
    const counts = { all: photos.length };
    for (const p of photos) counts[p.state] = (counts[p.state] || 0) + 1;
    return STATE_ORDER.filter((s) => counts[s]).map((s) => [s, `${s} ${counts[s]}`]);
  }, [photos, decisionMode, albumView, decisionOf]);

  const visibleIds = () => app.photos.filter(visibleFilter(app)).map((p) => p.id);
  const selectAll = () => setSelected(new Set(visibleIds()));
  const clear = () => setSelected(new Set());

  const title = SHARE ? SHARE.album : "PhotoRAW review";

  return (
    <header>
      <h1>{SHARE ? title : <>Photo<span>RAW</span> review</>}</h1>

      {!SHARE && (
        <select className="op-tool" value={albumView?.id || 0}
          onChange={(e) => viewAlbum(+e.target.value)} title="view an album">
          <option value={0}>All photos</option>
          {albums.map((a) => <option key={a.id} value={a.id}>{a.name} ({a.count})</option>)}
        </select>
      )}

      <div className="chips">
        {chips.map(([f, label]) => (
          <span key={f} className={"chip" + (filter === f ? " on" : "")} onClick={() => setFilter(f)}>{label}</span>
        ))}
      </div>

      <input id="search" type="search" placeholder="filename or #id…" value={search}
        onChange={(e) => setSearch(e.target.value)} />

      <span className="spacer" />

      {!SHARE && (
        <button className="hamburger" onClick={() => setMenuOpen((v) => !v)}>☰</button>
      )}

      <div className={"header-tools" + (menuOpen ? " open" : "")} style={SHARE ? {} : undefined}>
        {!SHARE && !albumView && (
          <span className="userchip" style={{ marginRight: 4 }}>
            {selected.size > 0 && <span>{selected.size} selected</span>}
          </span>
        )}
        {!SHARE && (
          <>
            <button className="ghost" onClick={selectAll}>Select all</button>
            <button className="ghost" onClick={clear}>Clear</button>
            <button onClick={() => app.setModal("upload")}>Upload…</button>
            <button onClick={() => app.setModal("albums")}>Albums…</button>
            {authEnabled && isSuperAdmin && <button onClick={() => app.setModal("users")}>Users…</button>}
          </>
        )}
        {(!SHARE || SHARE.permission === "develop") && (
          <button onClick={() => app.setJobsOpen((v) => !v)}>
            Jobs{app.jobs.some((j) => j.state === "queued" || j.state === "running") && <span className="live" />}
          </button>
        )}
        {!SHARE && (
          <button className="primary" disabled={!selected.size}
            onClick={() => app.openWizard([...selected].sort((a, b) => a - b))}>
            {selected.size > 1 ? `Redo ${selected.size} photos…` : "Redo selected…"}
          </button>
        )}
        {!SHARE && authEnabled && me && (
          <span className="userchip">
            <b>{me.name || me.username}</b>
            {(me.roles || []).map((r) => <span key={r} className="roletag">{r.replace("_", " ")}</span>)}
            <button className="ghost" onClick={async () => { await api.post("/auth/logout", {}); location.reload(); }}>Sign out</button>
          </span>
        )}
      </div>
    </header>
  );
}

// shared visibility predicate (also used by Gallery)
export function visibleFilter(app) {
  const q = app.search.trim().toLowerCase();
  return (p) => {
    if (app.albumView && !app.albumView.ids.has(p.id)) return false;
    if (app.decisionMode) {
      const d = app.decisionOf(p);
      if (app.filter === "sel" && d !== 1) return false;
      if (app.filter === "dis" && d !== 0) return false;
      if (app.filter === "und" && d !== null) return false;
    } else if (app.filter !== "all" && p.state !== app.filter) return false;
    if (q && !(p.filename.toLowerCase().includes(q) || ("#" + p.id).includes(q) || String(p.id) === q)) return false;
    return true;
  };
}
