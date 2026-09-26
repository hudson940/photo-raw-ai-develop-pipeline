import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AppCtx } from "./ctx.js";
import { api, SHARE, SHARE_THEME, downloadUrl, zipUrl, triggerDownload, saveOutput, fileToJpegBlob } from "./api.js";
import Login from "./components/Login.jsx";
import ShareLogin from "./components/ShareLogin.jsx";
import ShareCover from "./components/ShareCover.jsx";
import Header from "./components/Header.jsx";
import Gallery from "./components/Gallery.jsx";
import Lightbox from "./components/Lightbox.jsx";
import Wizard from "./components/Wizard.jsx";
import Albums from "./components/Albums.jsx";
import Users from "./components/Users.jsx";
import Jobs from "./components/Jobs.jsx";
import Upload from "./components/Upload.jsx";

// Apply the album presentation theme (share pages only) to <body>, and lazy-load the
// display fonts the quince años theme uses. Runs once at module load.
if (SHARE) {
  document.body.classList.add("theme-" + SHARE_THEME);
  if (SHARE_THEME === "quince") {
    const l = document.createElement("link");
    l.rel = "stylesheet";
    l.href = "https://fonts.googleapis.com/css2?family=Great+Vibes&family=Playfair+Display:ital,wght@0,500;0,600;1,500&family=Montserrat:wght@400;500;600&display=swap";
    document.head.appendChild(l);
  }
}

export default function App() {
  const [me, setMe] = useState(null);
  const [authEnabled, setAuthEnabled] = useState(false);
  const [ready, setReady] = useState(false);        // boot complete (auth resolved)
  const [needLogin, setNeedLogin] = useState(false);

  const [photos, setPhotos] = useState([]);
  const [selected, setSelected] = useState(() => new Set());
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState(() => {         // {key:'number'|'name'|'date', dir:1|-1}
    try { return JSON.parse(localStorage.getItem("pr_sort")) || { key: "number", dir: 1 }; }
    catch { return { key: "number", dir: 1 }; }
  });
  // download size/quality tier: high|print|medium|small (see api.downloadUrl)
  const [downloadPreset, setDownloadPreset] = useState(() => {
    try { return localStorage.getItem("pr_dlpreset") || "high"; } catch { return "high"; }
  });
  const [albumView, setAlbumView] = useState(null); // {id,name,ids:Set,dec:Map}
  const [albums, setAlbums] = useState([]);
  const [defaults, setDefaults] = useState(null);
  const [jobs, setJobs] = useState([]);

  const [lightbox, setLightbox] = useState(null);    // photo id
  const [wizardIds, setWizardIds] = useState(null);   // number[] | null
  const [modal, setModal] = useState(null);           // 'albums' | 'users' | 'upload' | null
  const [jobsOpen, setJobsOpen] = useState(false);
  const lastDoneSig = useRef("");

  const isSuperAdmin = !!(me && (me.roles || []).includes("super_admin"));

  const loadPhotos = useCallback(async () => {
    try { setPhotos((await api.get("/api/photos")).photos || []); } catch { /* handled by auth */ }
  }, []);

  const loadAlbums = useCallback(async () => {
    if (SHARE) return;
    try { setAlbums((await api.get("/api/albums")).albums || []); } catch { setAlbums([]); }
  }, []);

  const viewAlbum = useCallback(async (id) => {
    if (!id) { setAlbumView(null); setFilter("all"); return; }
    const d = await api.get(`/api/albums/${id}`);
    setAlbumView({
      id: d.id, name: d.name,
      ids: new Set(d.photos.map((p) => p.photo_id)),
      dec: new Map(d.photos.map((p) => [p.photo_id, p.decision])),
    });
    setFilter("all");
  }, []);

  const refreshJobs = useCallback(async () => {
    if (SHARE && SHARE.permission !== "develop") return;
    let list = [];
    try { list = (await api.get("/api/jobs")).jobs || []; } catch { return; }
    setJobs(list);
    const sig = list.map((j) => Object.values(j.items).filter((i) => i.status === "done").length).join(",");
    if (sig !== lastDoneSig.current) { lastDoneSig.current = sig; loadPhotos(); }
  }, [loadPhotos]);

  // boot: resolve auth, then load data
  useEffect(() => {
    (async () => {
      if (SHARE) { setReady(true); if (SHARE.authed) await loadPhotos(); return; }
      let info;
      try { info = await api.get("/api/me"); } catch { info = { authenticated: false, auth: false }; }
      setAuthEnabled(!!info.auth);
      if (info.authenticated) {
        setMe(info.user); setReady(true);
        loadPhotos(); loadAlbums(); refreshJobs();
      } else if (info.auth) {
        setNeedLogin(true); setReady(true);
      } else {
        setReady(true); loadPhotos(); loadAlbums(); refreshJobs();
      }
    })();
  }, [loadPhotos, loadAlbums, refreshJobs]);

  // poll jobs while any is active or the drawer is open
  useEffect(() => {
    if (!ready || needLogin) return;
    const active = jobs.some((j) => j.state === "queued" || j.state === "running");
    if (!active && !jobsOpen) return;
    const t = setTimeout(refreshJobs, 2500);
    return () => clearTimeout(t);
  }, [jobs, jobsOpen, ready, needLogin, refreshJobs]);

  // live-refresh album decisions while viewing one
  useEffect(() => {
    if (!albumView || SHARE) return;
    const t = setInterval(async () => {
      try {
        const d = await api.get(`/api/albums/${albumView.id}`);
        setAlbumView((av) => av && av.id === d.id ? {
          ...av, ids: new Set(d.photos.map((p) => p.photo_id)),
          dec: new Map(d.photos.map((p) => [p.photo_id, p.decision])),
        } : av);
      } catch {}
    }, 10000);
    return () => clearInterval(t);
  }, [albumView?.id]);

  const decisionMode = !!SHARE || !!albumView;
  const decisionOf = useCallback((p) => {
    if (SHARE) return p.decision ?? null;
    if (albumView) { const d = albumView.dec.get(p.id); return d === undefined ? null : d; }
    return null;
  }, [albumView]);

  const patchPhoto = useCallback((id, patch) => {
    setPhotos((ps) => ps.map((p) => (p.id === id ? { ...p, ...patch } : p)));
  }, []);

  const postDecision = useCallback(async (id, decision) => {
    try {
      await api.post("/api/decision", { photo_id: id, decision });
      if (SHARE) patchPhoto(id, { decision: decision === "select" ? 1 : decision === "discard" ? 0 : null });
      else setAlbumView((av) => {
        if (!av) return av;
        const dec = new Map(av.dec);
        dec.set(id, decision === "select" ? 1 : decision === "discard" ? 0 : null);
        return { ...av, dec };
      });
    } catch (e) { alert("Could not save your choice: " + e.message); }
  }, [patchPhoto]);

  const operatorSelect = useCallback(async (id, kind) => {
    try {
      const r = await api.post(`/api/photos/${id}/select`, { selected: kind });
      patchPhoto(id, { selected: r.selected });
    } catch (e) { alert("Could not update selection: " + e.message); }
  }, [patchPhoto]);

  const forceProcess = useCallback(async (id) => {
    try { await api.post(`/api/photos/${id}/process`, {}); setJobsOpen(true); refreshJobs(); }
    catch (e) { alert("Could not process: " + e.message); }
  }, [refreshJobs]);

  const setDeliver = useCallback(async (id, deliver) => {
    try {
      await api.post(`/api/photos/${id}/deliver`, { deliver });
      patchPhoto(id, { deliver });
    } catch (e) { alert("Could not change the delivered version: " + e.message); }
  }, [patchPhoto]);

  // operator: replace a photo's edited version with an uploaded image (keeps the name)
  const replacePhoto = useCallback(async (id, file) => {
    const blob = await fileToJpegBlob(file);
    const r = await saveOutput(id, blob);
    patchPhoto(id, { output_mtime: r.output_mtime, has_output: true });
    return r;
  }, [patchPhoto]);

  const deletePhoto = useCallback(async (id) => {
    await api.post(`/api/photos/${id}/delete`, {});
    setPhotos((ps) => ps.filter((p) => p.id !== id));
    setSelected((s) => { const n = new Set(s); n.delete(id); return n; });
  }, []);

  // bulk delete; drops whatever the server actually deleted from the list + selection
  const deletePhotos = useCallback(async (ids) => {
    if (!ids || !ids.length) return;
    const r = await api.post("/api/photos/delete", { ids });
    const gone = new Set(r.deleted || []);
    setPhotos((ps) => ps.filter((p) => !gone.has(p.id)));
    setSelected((s) => new Set([...s].filter((id) => !gone.has(id))));
    return r;
  }, []);

  // download one photo, or a zip of many (each honoring its developed/original choice)
  const downloadPhotos = useCallback((ids) => {
    if (!ids || !ids.length) return;
    triggerDownload(ids.length === 1
      ? downloadUrl(ids[0], downloadPreset)
      : zipUrl([...ids].sort((a, b) => a - b), downloadPreset));
  }, [downloadPreset]);

  // remember the chosen download tier across reloads
  const changeDownloadPreset = useCallback((p) => {
    setDownloadPreset(p);
    try { localStorage.setItem("pr_dlpreset", p); } catch {}
  }, []);

  // persist the chosen photo order so it sticks across reloads
  const changeSort = useCallback((patch) => {
    setSort((s) => {
      const n = { ...s, ...patch };
      try { localStorage.setItem("pr_sort", JSON.stringify(n)); } catch {}
      return n;
    });
  }, []);

  const openWizard = useCallback((ids) => setWizardIds(ids), []);
  const ensureDefaults = useCallback(async () => {
    if (defaults) return defaults;
    const d = (await api.get("/api/defaults")).retouch;
    setDefaults(d); return d;
  }, [defaults]);

  const value = {
    me, authEnabled, isSuperAdmin, share: SHARE,
    photos, loadPhotos, patchPhoto,
    selected, setSelected,
    filter, setFilter, search, setSearch, sort, changeSort,
    albums, loadAlbums, albumView, viewAlbum,
    jobs, refreshJobs, jobsOpen, setJobsOpen,
    lightbox, setLightbox,
    wizardIds, openWizard, closeWizard: () => setWizardIds(null),
    modal, setModal,
    decisionMode, decisionOf, postDecision, operatorSelect, forceProcess,
    setDeliver, downloadPhotos, downloadPreset, changeDownloadPreset, deletePhoto, deletePhotos, replacePhoto,
    defaults, ensureDefaults,
    theme: SHARE_THEME,
  };

  if (!ready) return <div className="empty" style={{ marginTop: 80 }}>Loading…</div>;
  if (needLogin) return <Login />;
  if (SHARE && SHARE.invalid)
    return <div className="empty" style={{ marginTop: 80 }}>This album link is no longer valid.</div>;
  if (SHARE && !SHARE.authed) return <ShareLogin />;

  const themed = SHARE && SHARE_THEME !== "default";
  return (
    <AppCtx.Provider value={value}>
      <Header />
      {themed && <ShareCover />}
      <Gallery />
      {lightbox != null && <Lightbox />}
      {wizardIds && <Wizard />}
      {modal === "albums" && <Albums />}
      {modal === "users" && <Users />}
      {modal === "upload" && <Upload />}
      {jobsOpen && <Jobs />}
    </AppCtx.Provider>
  );
}
