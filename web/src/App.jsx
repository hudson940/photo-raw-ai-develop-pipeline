import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AppCtx } from "./ctx.js";
import { api, SHARE } from "./api.js";
import Login from "./components/Login.jsx";
import Header from "./components/Header.jsx";
import Gallery from "./components/Gallery.jsx";
import Lightbox from "./components/Lightbox.jsx";
import Wizard from "./components/Wizard.jsx";
import Albums from "./components/Albums.jsx";
import Users from "./components/Users.jsx";
import Jobs from "./components/Jobs.jsx";
import Upload from "./components/Upload.jsx";

export default function App() {
  const [me, setMe] = useState(null);
  const [authEnabled, setAuthEnabled] = useState(false);
  const [ready, setReady] = useState(false);        // boot complete (auth resolved)
  const [needLogin, setNeedLogin] = useState(false);

  const [photos, setPhotos] = useState([]);
  const [selected, setSelected] = useState(() => new Set());
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
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
      if (SHARE) { setReady(true); await loadPhotos(); return; }
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
    filter, setFilter, search, setSearch,
    albums, loadAlbums, albumView, viewAlbum,
    jobs, refreshJobs, jobsOpen, setJobsOpen,
    lightbox, setLightbox,
    wizardIds, openWizard, closeWizard: () => setWizardIds(null),
    modal, setModal,
    decisionMode, decisionOf, postDecision, operatorSelect, forceProcess,
    defaults, ensureDefaults,
  };

  if (!ready) return <div className="empty" style={{ marginTop: 80 }}>Loading…</div>;
  if (needLogin) return <Login />;

  return (
    <AppCtx.Provider value={value}>
      <Header />
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
