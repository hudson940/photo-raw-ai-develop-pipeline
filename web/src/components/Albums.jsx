import React, { useEffect, useState } from "react";
import { useApp } from "../ctx.js";
import { api } from "../api.js";

export const THEMES = [["default", "Default"], ["quince", "Quince años"]];

export default function Albums() {
  const app = useApp();
  const [albums, setAlbums] = useState([]);
  const [name, setName] = useState("");
  const [theme, setTheme] = useState("default");
  const [err, setErr] = useState("");
  const nsel = app.selected.size;

  const load = async () => { try { setAlbums((await api.get("/api/albums")).albums || []); } catch { setAlbums([]); } };
  useEffect(() => { load(); }, []);

  const create = async () => {
    setErr("");
    try {
      await api.post("/api/albums", { name: name.trim() || `Album ${new Date().toISOString().slice(0, 10)}`, theme, photo_ids: [...app.selected] });
      setName(""); load(); app.loadAlbums();
    } catch (e) { setErr(e.message); }
  };

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && app.setModal(null)}>
      <div className="modal">
        <div className="head"><h2>Albums</h2><span className="spacer" /><button className="ghost" onClick={() => app.setModal(null)}>✕</button></div>
        <div className="body">
          <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
            <input type="text" placeholder="new album name" value={name} onChange={(e) => setName(e.target.value)} style={{ flex: 1, minWidth: 140 }} />
            <select value={theme} onChange={(e) => setTheme(e.target.value)} title="album presentation theme">
              {THEMES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
            <button className="primary" disabled={!nsel} onClick={create} title="select photos first">Create with {nsel} selected</button>
          </div>
          {err && <div className="err" style={{ marginBottom: 10 }}>{err}</div>}
          {albums.length ? albums.map((a) => <AlbumRow key={a.id} a={a} nsel={nsel} reload={() => { load(); app.loadAlbums(); }} />)
            : <div className="hint">No albums yet — select photos, then create one here.</div>}
        </div>
      </div>
    </div>
  );
}

function AlbumRow({ a, nsel, reload }) {
  const app = useApp();
  const [pw, setPw] = useState("");
  const [perm, setPerm] = useState("select");
  const [err, setErr] = useState("");

  const addSelected = async () => { await api.post(`/api/albums/${a.id}/photos`, { add: [...app.selected] }); reload(); if (app.albumView?.id === a.id) app.viewAlbum(a.id); };
  const del = async () => { if (!confirm(`Delete album "${a.name}"? (photos are kept)`)) return; await api.post(`/api/albums/${a.id}/delete`, {}); if (app.albumView?.id === a.id) app.viewAlbum(0); reload(); };
  const finalize = async () => {
    if (!confirm(`Finalize "${a.name}"?\n\nThis PERMANENTLY deletes the RAW files for all ${a.count} photo(s) — from disk and cloud storage — keeping only the developed images. RAWs are shared, so any photo also in another album loses its RAW too. This cannot be undone and the photos can no longer be re-developed.`)) return;
    setErr("");
    try { const r = await api.post(`/api/albums/${a.id}/finalize`, {}); reload(); if (app.albumView?.id === a.id) app.loadPhotos(); alert(`Finalized: removed ${r.freed_raws} RAW file(s).`); }
    catch (e) { setErr(e.message); }
  };
  const createShare = async () => {
    setErr(""); if (pw.length < 4) { setErr("password must be at least 4 characters"); return; }
    try { await api.post(`/api/albums/${a.id}/shares`, { password: pw, permission: perm }); setPw(""); reload(); }
    catch (e) { setErr(e.message); }
  };
  const revoke = async (t) => { if (!confirm("Revoke this link?")) return; await api.post(`/api/shares/${t}/revoke`, {}); reload(); };
  const copy = async (url) => { try { await navigator.clipboard.writeText(location.origin + url); } catch { prompt("Copy the link:", location.origin + url); } };
  const setTheme = async (t) => { await api.post(`/api/albums/${a.id}/theme`, { theme: t }); reload(); };

  return (
    <div className="albrow">
      <div className="row1">
        <b>{a.name}</b>
        <span className="muted" style={{ fontSize: 12.5 }}>{a.count} photos · ✓ {a.selected} · ✗ {a.discarded}</span>
        {a.finalized_at && <span className="tag" title={`RAWs removed on ${new Date(a.finalized_at * 1000).toLocaleDateString()}`}>finalized</span>}
        <span className="spacer" />
        <select value={a.theme || "default"} onChange={(e) => setTheme(e.target.value)} title="album theme">
          {THEMES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <button onClick={() => { app.setModal(null); app.viewAlbum(a.id); }}>View</button>
        <button disabled={!nsel} onClick={addSelected}>Add {nsel} selected</button>
        {!a.finalized_at && <button onClick={finalize} title="delete RAWs, keep final images (frees storage)">Finalize</button>}
        <button onClick={del}>Delete</button>
      </div>
      {a.shares.map((s) => (
        <div className="shline" key={s.token}>
          <span className="tag">{s.permission === "develop" ? "full develop" : "select only"}</span>
          <span className="lnk">{location.origin}{s.url}</span>
          <button onClick={() => copy(s.url)}>Copy link</button>
          <button onClick={() => revoke(s.token)}>Revoke</button>
        </div>
      ))}
      <div className="shline">
        <input type="password" placeholder="link password (min 4)" value={pw} onChange={(e) => setPw(e.target.value)} style={{ width: 170 }} />
        <select value={perm} onChange={(e) => setPerm(e.target.value)}>
          <option value="select">select &amp; discard only</option>
          <option value="develop">full develop</option>
        </select>
        <button onClick={createShare}>Create share link</button>
        {err && <span className="err">{err}</span>}
      </div>
    </div>
  );
}
