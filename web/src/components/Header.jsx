import React, { useMemo, useState } from "react";
import { useApp } from "../ctx.js";
import { api, SHARE } from "../api.js";

const STATE_ORDER = ["all", "done", "retouched", "developed", "analyzed", "previewed", "pending", "rejected", "review", "failed"];

export default function Header() {
  const app = useApp();
  const { photos, filter, setFilter, search, setSearch, selected, setSelected,
    decisionMode, decisionOf, albumView, viewAlbum, albums, me, authEnabled, isSuperAdmin } = app;
  const [menuOpen, setMenuOpen] = useState(false);

  const deleteSelected = async () => {
    const ids = [...selected];
    if (!ids.length) return;
    const what = ids.length === 1 ? "this photo" : `these ${ids.length} photos`;
    if (!confirm(`Delete ${what}?\nThis permanently removes them and all their files.`)) return;
    try {
      const r = await app.deletePhotos(ids);
      if (r?.missing?.length) alert(`${r.missing.length} photo(s) could not be deleted (not found or not yours).`);
    } catch (e) { alert("Delete failed: " + e.message); }
  };

  const chips = useMemo(() => {
    if (decisionMode) {
      const pool = photos.filter((p) => !albumView || albumView.ids.has(p.id));
      const sel = pool.filter((p) => decisionOf(p) === 1).length;
      const dis = pool.filter((p) => decisionOf(p) === 0).length;
      // "all" hides discarded — it's selected + undecided; use "dis" to see discarded
      return [["all", `all ${pool.length - dis}`], ["sel", `✓ selected ${sel}`],
        ["und", `undecided ${pool.length - sel - dis}`], ["dis", `✗ discarded ${dis}`]];
    }
    const counts = {};
    let cut = 0;
    for (const p of photos) {
      if (p.selected === 0) { cut++; continue; }   // discarded — counted separately, hidden by default
      counts[p.state] = (counts[p.state] || 0) + 1;
    }
    const list = [["all", `all ${photos.length - cut}`],
      ...STATE_ORDER.filter((s) => s !== "all" && counts[s]).map((s) => [s, `${s} ${counts[s]}`])];
    if (cut) list.push(["cut", `✗ discarded ${cut}`]);
    return list;
  }, [photos, decisionMode, albumView, decisionOf]);

  const visibleIds = () => app.photos.filter(visibleFilter(app)).map((p) => p.id);
  const selectAll = () => setSelected(new Set(visibleIds()));
  const clear = () => setSelected(new Set());

  // photos a share customer marked ✓ — the set they'd want to download
  const chosenShareIds = useMemo(
    () => (SHARE ? photos.filter((p) => decisionOf(p) === 1).map((p) => p.id) : []),
    [SHARE, photos, decisionOf]);

  const title = SHARE ? SHARE.album : "PhotoCapibara review";

  return (
    <header>
      <h1>
        <img className="logo" src="/logo.png" alt="PhotoCapibara" />
        {SHARE ? title : <>Photo<span>Capibara</span> review</>}
      </h1>

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

      <span className="sortgrp" title="order photos">
        <select value={app.sort.key} onChange={(e) => app.changeSort({ key: e.target.value })}>
          <option value="number">Number</option>
          <option value="name">Name</option>
          <option value="date">Date</option>
        </select>
        <button className="ghost" title={app.sort.dir === 1 ? "ascending" : "descending"}
          onClick={() => app.changeSort({ dir: app.sort.dir === 1 ? -1 : 1 })}>
          {app.sort.dir === 1 ? "▲" : "▼"}
        </button>
      </span>

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
        <span className="qualgrp" title="download size / quality">
          <select value={app.downloadPreset} onChange={(e) => app.changeDownloadPreset(e.target.value)}>
            <option value="high">High quality</option>
            <option value="print">Print (A4)</option>
            <option value="medium">Medium</option>
            <option value="small">Small</option>
          </select>
        </span>
        {!SHARE && (
          <>
            <button className="ghost" onClick={selectAll}>Select all</button>
            <button className="ghost" onClick={clear}>Clear</button>
            <button disabled={!selected.size} onClick={() => app.downloadPhotos([...selected])}>
              {selected.size > 1 ? `Download ${selected.size}` : "Download"}
            </button>
            <button className="danger" disabled={!selected.size} onClick={deleteSelected}>
              {selected.size > 1 ? `Delete ${selected.size}` : "Delete"}
            </button>
            <button onClick={() => app.setModal("upload")}>Upload…</button>
            <button onClick={() => app.setModal("albums")}>Albums…</button>
            {authEnabled && isSuperAdmin && <button onClick={() => app.setModal("users")}>Users…</button>}
          </>
        )}
        {SHARE && (
          <button disabled={!chosenShareIds.length}
            onClick={() => app.downloadPhotos(chosenShareIds)}>
            {chosenShareIds.length > 1 ? `Download ${chosenShareIds.length} selected` : "Download selected"}
          </button>
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
  // which filter chip reveals discarded photos in the current mode
  const discardedFilter = app.decisionMode ? "dis" : "cut";
  return (p) => {
    if (app.albumView && !app.albumView.ids.has(p.id)) return false;
    const discarded = app.decisionMode ? app.decisionOf(p) === 0 : p.selected === 0;
    // hide discarded photos everywhere except the explicit discarded filter
    if (discarded && app.filter !== discardedFilter) return false;
    if (app.decisionMode) {
      const d = app.decisionOf(p);
      if (app.filter === "sel" && d !== 1) return false;
      if (app.filter === "dis" && d !== 0) return false;
      if (app.filter === "und" && d !== null) return false;
    } else if (app.filter === "cut") {
      if (!discarded) return false;
    } else if (app.filter !== "all" && p.state !== app.filter) return false;
    if (q && !(p.filename.toLowerCase().includes(q) || ("#" + p.id).includes(q) || String(p.id) === q)) return false;
    return true;
  };
}

// shared sort comparator (Gallery + Lightbox). Order by number (#id), name, or date;
// direction 1 = ascending, -1 = descending. Ties break on id for a stable order.
export function sortComparator(app) {
  const { key, dir } = app.sort || { key: "number", dir: 1 };
  return (a, b) => {
    let r;
    if (key === "name") r = a.filename.localeCompare(b.filename, undefined, { numeric: true, sensitivity: "base" });
    else if (key === "date") r = (a.created_at || 0) - (b.created_at || 0);
    else r = a.id - b.id;                 // number
    if (r === 0) r = a.id - b.id;
    return r * dir;
  };
}
