import React, { useEffect, useMemo, useState } from "react";
import { useApp, fmtIds } from "../ctx.js";
import { api } from "../api.js";
import { SECTIONS, pkey, currentValue, buildOverrides, describeOverrides, fmtV } from "../redoParams.js";
import { ParamRow, RedoOptions, DEFAULT_OPTS, redoBody } from "./ParamControls.jsx";

const STEPS = [...SECTIONS, { title: "Review", review: true }];

export default function Wizard() {
  const app = useApp();
  const ids = app.wizardIds;
  const [step, setStep] = useState(0);
  const [detail, setDetail] = useState(null);
  const [defs, setDefs] = useState(null);
  const [enabled, setEnabled] = useState({});     // "g.k" -> bool
  const [values, setValues] = useState({});       // "g.k" -> value
  const [opts, setOpts] = useState(DEFAULT_OPTS);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    (async () => {
      setDefs(await app.ensureDefaults());
      if (ids.length === 1) {
        try { const d = await api.get(`/api/photos/${ids[0]}`); if (d.has_analysis) setDetail(d); } catch {}
      }
    })();
  }, []);

  const valOf = (prm) => (pkey(prm) in values ? values[pkey(prm)] : currentValue(prm, detail, defs));
  // ticking a row applies the value its control shows
  const toggle = (prm, v) => {
    const k = pkey(prm);
    if (v && !(k in values)) setValues((s) => ({ ...s, [k]: valOf(prm) }));
    setEnabled((s) => ({ ...s, [k]: v }));
  };
  const setVal = (prm, v) => { setValues((s) => ({ ...s, [pkey(prm)]: v })); setEnabled((s) => ({ ...s, [pkey(prm)]: true })); };
  const shown = (prm) => !prm.show || prm.show(Object.fromEntries(SECTIONS.flatMap((s) => s.params).map((p) => [pkey(p), valOf(p)])));
  const overrides = useMemo(() => buildOverrides(enabled, values), [enabled, values]);

  const run = async () => {
    setBusy(true); setMsg("");
    try {
      await api.post("/api/redo", redoBody(ids, overrides, opts));
      app.closeWizard(); app.setJobsOpen(true); app.refreshJobs();
    } catch (e) { setMsg("Error: " + e.message); }
    finally { setBusy(false); }
  };

  const last = step === STEPS.length - 1;
  const s = STEPS[step];
  const stepActive = (i) => (STEPS[i].params || []).some((prm) => enabled[pkey(prm)]);
  const entries = describeOverrides(overrides);

  return (
    <div className="overlay wizard-overlay" onMouseDown={(e) => e.target === e.currentTarget && app.closeWizard()}>
      <div className="modal" style={{ width: "min(880px,96vw)" }}>
        <div className="head">
          <h2>Redo wizard</h2>
          <span className="muted">{ids.length === 1 ? `photo #${ids[0]}` : `${ids.length} photos: ${fmtIds(ids)}`}</span>
          <span className="spacer" /><button className="ghost" onClick={app.closeWizard}>✕</button>
        </div>
        <div className="tabs">
          {STEPS.map((st, i) => (
            <span key={i} className={"tab" + (i === step ? " on" : "")} onClick={() => setStep(i)}>
              {st.title}{!st.review && stepActive(i) && <span className="dot" />}
            </span>
          ))}
        </div>
        <div className="body" style={{ background: "var(--panel2)", minHeight: 300 }}>
          {s.review ? (
            <>
              <div className="hint">Only enabled parameters are overridden — everything else keeps each photo's current values.</div>
              <div className="sect">Photos ({ids.length})</div>
              <div>{fmtIds(ids)}</div>
              <div className="sect">Overrides ({entries.length})</div>
              {entries.length ? entries.map(([k, v]) => <div key={k}>• <b>{k}</b> → {String(fmtV(v))}</div>)
                : <div className="muted" style={{ color: "var(--warn)" }}>None — re-renders with current params.</div>}
              <div className="sect" style={{ marginTop: 14 }}>Options</div>
              <RedoOptions opts={opts} setOpts={setOpts} />
            </>
          ) : (
            <>
              {s.hint && <div className="hint">{s.hint}</div>}
              {s.params.filter(shown).map((prm) => (
                <ParamRow key={pkey(prm)} prm={prm} on={!!enabled[pkey(prm)]} value={valOf(prm)} showNow={!!detail}
                  toggle={(v) => toggle(prm, v)} setVal={(v) => setVal(prm, v)} />
              ))}
            </>
          )}
        </div>
        <div className="foot">
          <span className="muted" style={{ fontSize: 12.5 }}>{msg}</span>
          <span className="spacer" />
          {step > 0 && <button onClick={() => setStep(step - 1)}>Back</button>}
          {!last && <button className="primary" onClick={() => setStep(step + 1)}>Next</button>}
          {last && <button className="primary" disabled={busy} onClick={run}>{busy ? "Queuing…" : "Run batch"}</button>}
        </div>
      </div>
    </div>
  );
}
