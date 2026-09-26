import React, { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../ctx.js";
import { api, SHARE } from "../api.js";
import { SECTIONS, pkey, currentValue, buildOverrides, describeOverrides, fmtV } from "../redoParams.js";
import { ParamRow, RedoOptions, DEFAULT_OPTS, redoBody } from "./ParamControls.jsx";

// Right-hand panel of the photo editor: every redo option for the open photo, grouped in
// collapsible sections. Checked rows are sent as overrides; Render queues a one-photo redo
// and the new image appears when the job finishes.
export default function AdjustPanel({ photo, rendering }) {
  const app = useApp();
  const [detail, setDetail] = useState(null);
  const [defs, setDefs] = useState(null);
  const [enabled, setEnabled] = useState({});
  const [values, setValues] = useState({});
  const [opts, setOpts] = useState(DEFAULT_OPTS);
  const [open, setOpen] = useState(() => {
    try { return new Set(JSON.parse(localStorage.getItem("pr_adjust_open")) || ["light", "background"]); }
    catch { return new Set(["light", "background"]); }
  });
  const [info, setInfo] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const wasRendering = useRef(rendering);

  const load = async () => {
    try { setDetail(await api.get(`/api/photos/${photo.id}`)); } catch { setDetail(null); }
  };
  const reset = () => { setEnabled({}); setValues({}); setOpts(DEFAULT_OPTS); setMsg(""); };

  useEffect(() => { app.ensureDefaults().then(setDefs).catch(() => {}); }, []);
  useEffect(() => { reset(); setDetail(null); load(); }, [photo.id]);
  // a render of this photo just finished: its saved params changed, reload them
  useEffect(() => {
    if (wasRendering.current && !rendering) { reset(); load(); setMsg("Render finished."); }
    wasRendering.current = rendering;
  }, [rendering]);

  const toggleSection = (id) => setOpen((s) => {
    const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id);
    try { localStorage.setItem("pr_adjust_open", JSON.stringify([...n])); } catch {}
    return n;
  });

  const valOf = (prm) => (pkey(prm) in values ? values[pkey(prm)] : currentValue(prm, detail, defs));
  // ticking a row applies the value its control shows
  const toggle = (prm, v) => {
    const k = pkey(prm);
    if (v && !(k in values)) setValues((s) => ({ ...s, [k]: valOf(prm) }));
    setEnabled((s) => ({ ...s, [k]: v }));
  };
  const setVal = (prm, v) => { setValues((s) => ({ ...s, [pkey(prm)]: v })); setEnabled((s) => ({ ...s, [pkey(prm)]: true })); };
  const flat = useMemo(() => SECTIONS.flatMap((s) => s.params), []);
  const allVals = Object.fromEntries(flat.map((p) => [pkey(p), valOf(p)]));
  const overrides = useMemo(() => buildOverrides(enabled, values), [enabled, values]);
  const changes = describeOverrides(overrides);
  const nOn = (s) => s.params.filter((p) => enabled[pkey(p)]).length;

  const render = async () => {
    setBusy(true); setMsg("");
    // a redo without an explicit background action resets it to the default (auto); in the
    // editor, re-rendering must keep the background the photo was last rendered with
    const ov = { ...overrides, retouch: { ...(overrides.retouch || {}) } };
    const actionPrm = flat.find((p) => pkey(p) === "background.action");
    ov.retouch.background = { action: valOf(actionPrm) || "auto", ...(ov.retouch.background || {}) };
    try {
      await api.post("/api/redo", redoBody([photo.id], ov, opts));
      setMsg("Queued — rendering…");
      app.refreshJobs();
    } catch (e) { setMsg("Error: " + e.message); }
    finally { setBusy(false); }
  };

  return (
    <aside className="adjust">
      <div className="adj-head">
        <b>Adjust</b>
        <span className="muted">#{photo.id}</span>
        <span className="spacer" />
        <button className="ghost sm" onClick={() => setInfo(!info)} title="scene + reproduce command">{info ? "Hide info" : "Info"}</button>
        <button className="ghost sm" disabled={!changes.length} onClick={reset} title="discard unsaved changes">Reset</button>
      </div>

      <div className="adj-scroll">
        {info && detail && (
          <div className="adj-info">
            <div className="sect">Scene</div><div className="muted">{detail.scene || "—"}</div>
            {detail.command && <><div className="sect">Reproduce</div><div className="cmd">{detail.command}</div></>}
          </div>
        )}
        {!detail?.has_analysis && detail && (
          <div className="hint" style={{ padding: "0 12px" }}>Not analyzed yet — values shown are the defaults.</div>
        )}
        {SECTIONS.map((s) => {
          const isOpen = open.has(s.id);
          const n = nOn(s);
          return (
            <section key={s.id} className={"adj-sec" + (isOpen ? " open" : "")}>
              <header onClick={() => toggleSection(s.id)}>
                <span className="ico">{s.icon}</span>
                <span className="t">{s.title}</span>
                {n > 0 && <span className="cnt">{n}</span>}
                <span className="spacer" />
                <span className="chev">{isOpen ? "▾" : "▸"}</span>
              </header>
              {isOpen && (
                <div className="adj-body">
                  {s.hint && <div className="hint">{s.hint}</div>}
                  {s.id === "erase" && detail && !detail.retouch?.erase?.mask && (
                    <div className="hint warn">No saved mask — these settings do nothing until you paint the area with the <b>Erase</b> tool (left bar) and click Remove &amp; render.</div>
                  )}
                  {s.params.filter((prm) => !prm.show || prm.show(allVals)).map((prm) => (
                    <ParamRow key={pkey(prm)} prm={prm} on={!!enabled[pkey(prm)]} value={valOf(prm)}
                      toggle={(v) => toggle(prm, v)} setVal={(v) => setVal(prm, v)} />
                  ))}
                </div>
              )}
            </section>
          );
        })}
        <section className="adj-sec open">
          <header><span className="ico">⚙</span><span className="t">Render options</span></header>
          <div className="adj-body"><RedoOptions opts={opts} setOpts={setOpts} /></div>
        </section>
      </div>

      <div className="adj-foot">
        <div className="adj-changes">
          {changes.length
            ? changes.map(([k, v]) => <span key={k} className="pill">{k} → {String(fmtV(v))}</span>)
            : <span className="muted">No changes — Render re-renders with the current settings.</span>}
        </div>
        <div className="row">
          <span className="muted status">{rendering ? <><span className="live" /> rendering…</> : msg}</span>
          <span className="spacer" />
          {!SHARE && <button onClick={() => app.openWizard([photo.id])} title="open the step-by-step wizard">Wizard…</button>}
          <button className="primary" disabled={busy || rendering} onClick={render}>{busy ? "Queuing…" : "Render"}</button>
        </div>
      </div>
    </aside>
  );
}
