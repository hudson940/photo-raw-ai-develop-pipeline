import React, { useEffect, useMemo, useState } from "react";
import { useApp, fmtIds } from "../ctx.js";
import { api, SHARE } from "../api.js";

const SKIN_TYPES = ["auto", "fair", "light", "medium", "olive", "tan", "brown", "deep"];
const BG_COLORS = ["gray", "white", "black", "charcoal", "blue", "navy", "teal", "green", "red", "maroon", "pink", "purple", "beige", "brown"];

const STEPS = [
  { title: "Skin", hint: "Tone correction targets natural best-practice colour per skin type. Warmth/richness/red ride on tone correction.", params: [
    { g: "retouch", k: "skin_smoothing", label: "Skin smoothing", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "remove_blemishes", label: "Remove blemishes", type: "bool" },
    { g: "retouch", k: "skin_tone_correction", label: "Tone correction", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "skin_type", label: "Skin type", type: "select", options: SKIN_TYPES },
    { g: "retouch", k: "skin_warmth", label: "Warmth (yellow↔orange)", min: -1, max: 1, step: 0.05 },
    { g: "retouch", k: "skin_saturation", label: "Richness / saturation", min: 0.5, max: 1.8, step: 0.05 },
    { g: "retouch", k: "skin_luminance", label: "Skin luminance", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "skin_red", label: "Red flush (cheeks)", min: 0, max: 1, step: 0.05 },
  ] },
  { title: "Face", params: [
    { g: "retouch", k: "brighten_eyes", label: "Brighten eyes", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "iris_enhance", label: "Iris pop", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "whiten_teeth", label: "Whiten teeth", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "lip_enhance", label: "Lip enhancement", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "reduce_dark_circles", label: "Dark circles", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "reduce_dewlap", label: "Dewlap / double chin", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "tame_highlights", label: "Tame shiny highlights", min: 0, max: 1, step: 0.05 },
  ] },
  { title: "Hair & clothing", params: [
    { g: "retouch", k: "hair_texture", label: "Hair texture", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "hair_shimmer", label: "Hair shimmer", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "hair_defrizz", label: "Defrizz", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "clothing_contrast", label: "Clothing contrast", min: 0, max: 1, step: 0.05 },
    { g: "clothing", k: "color", label: "Cloth colour", type: "text", ph: "red, navy…" },
    { g: "clothing", k: "color_pop", label: "Colour pop", min: 0, max: 1, step: 0.05 },
    { g: "clothing", k: "luminance", label: "Cloth luminance", min: -1, max: 1, step: 0.05 },
    { g: "clothing", k: "shadows", label: "Cloth shadows", min: -1, max: 1, step: 0.05 },
    { g: "clothing", k: "blacks", label: "Cloth blacks", min: -1, max: 1, step: 0.05 },
    { g: "clothing", k: "whites", label: "Cloth whites", min: -1, max: 1, step: 0.05 },
  ] },
  { title: "Light", hint: "White balance / shadows / highlights re-develop the RAW.", params: [
    { g: "wb", k: "wb", label: "White balance", type: "wb" },
    { g: "develop", k: "shadows", label: "Shadows (global)", min: -100, max: 100, step: 1 },
    { g: "develop", k: "highlights", label: "Highlights (global)", min: -100, max: 100, step: 1 },
    { g: "retouch", k: "subject_exposure", label: "Subject exposure (EV)", min: -2, max: 2, step: 0.1 },
    { g: "retouch", k: "background_exposure", label: "Background exposure (EV)", min: -2, max: 2, step: 0.1 },
    { g: "retouch", k: "auto_levels", label: "Enable auto-levels", type: "bool" },
    { g: "retouch", k: "auto_skin_target", label: "Auto skin target", min: 0, max: 1, step: 0.01 },
    { g: "retouch", k: "intensity", label: "Retouch intensity", type: "select", options: ["subtle", "natural", "polished"] },
  ] },
  { title: "Background", hint: "Background changes only when set here.", params: [
    { g: "background", k: "action", label: "Action", type: "select", options: ["auto", "keep", "blur", "smooth", "studio", "replace"] },
    { g: "background", k: "color", label: "Studio colour", type: "select", options: BG_COLORS },
    { g: "background", k: "replace_prompt", label: "Replace prompt", type: "text", ph: "soft window light…" },
  ] },
  { title: "Review", review: true },
];

export default function Wizard() {
  const app = useApp();
  const ids = app.wizardIds;
  const [step, setStep] = useState(0);
  const [detail, setDetail] = useState(null);
  const [defs, setDefs] = useState(null);
  const [enabled, setEnabled] = useState({});     // "g.k" -> bool
  const [values, setValues] = useState({});       // "g.k" -> value
  const [opts, setOpts] = useState({ from_raw: false, reanalyze: false, subject_only: false, skin_exposure: false });
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

  const cur = (prm) => {
    const src = detail, dfl = defs || {};
    if (prm.g === "retouch") return src ? src.retouch[prm.k] : dfl[prm.k];
    if (prm.g === "clothing") return src ? (src.retouch.clothing || {})[prm.k] : (dfl.clothing || {})[prm.k];
    if (prm.g === "background") return src ? (src.retouch.background || {})[prm.k] : (dfl.background || {})[prm.k];
    if (prm.g === "develop") return src ? (src.develop || {})[prm.k] : 0;
    if (prm.g === "wb") return src ? (src.develop?.white_balance || {}) : {};
    return undefined;
  };
  const key = (prm) => `${prm.g}.${prm.k}`;
  const setVal = (prm, v) => { setValues((s) => ({ ...s, [key(prm)]: v })); setEnabled((s) => ({ ...s, [key(prm)]: true })); };
  const valOf = (prm) => (key(prm) in values ? values[key(prm)] : cur(prm));

  const overrides = useMemo(() => {
    const ov = { retouch: {}, develop: {}, white_balance: {}, };
    for (const s of STEPS) for (const prm of s.params || []) {
      if (!enabled[key(prm)]) continue;
      const v = values[key(prm)];
      if (prm.g === "wb") { Object.assign(ov.white_balance, v); continue; }
      if (prm.g === "retouch") ov.retouch[prm.k] = v;
      else if (prm.g === "develop") ov.develop[prm.k] = v;
      else if (prm.g === "clothing") (ov.retouch.clothing ||= {})[prm.k] = v;
      else if (prm.g === "background") (ov.retouch.background ||= {})[prm.k] = v;
    }
    if (!Object.keys(ov.retouch).length) delete ov.retouch;
    if (!Object.keys(ov.develop).length) delete ov.develop;
    if (!Object.keys(ov.white_balance).length) delete ov.white_balance;
    return ov;
  }, [enabled, values]);

  const run = async () => {
    setBusy(true); setMsg("");
    try {
      await api.post("/api/redo", { ids, overrides, from_raw: opts.from_raw, reanalyze: opts.reanalyze,
        subject_only: opts.reanalyze && opts.subject_only, skin_exposure: opts.reanalyze && opts.skin_exposure });
      app.closeWizard(); app.setJobsOpen(true); app.refreshJobs();
    } catch (e) { setMsg("Error: " + e.message); }
    finally { setBusy(false); }
  };

  const last = step === STEPS.length - 1;
  const s = STEPS[step];
  const stepActive = (i) => (STEPS[i].params || []).some((prm) => enabled[key(prm)]);

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && app.closeWizard()}>
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
            <Review ids={ids} overrides={overrides} opts={opts} setOpts={setOpts} />
          ) : (
            <>
              {s.hint && <div className="hint">{s.hint}</div>}
              {s.params.map((prm) => (
                <ParamRow key={key(prm)} prm={prm} on={!!enabled[key(prm)]} value={valOf(prm)}
                  toggle={(v) => setEnabled((st2) => ({ ...st2, [key(prm)]: v }))} setVal={(v) => setVal(prm, v)} detail={detail} />
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

function ParamRow({ prm, on, value, toggle, setVal, detail }) {
  const nowLabel = detail ? `now: ${fmtV(value)}` : "";
  let ctl;
  if (prm.type === "bool") ctl = <label className="ctl"><input type="checkbox" checked={value === true} onChange={(e) => setVal(e.target.checked)} /> on</label>;
  else if (prm.type === "select") ctl = <span className="ctl"><select value={String(value ?? prm.options[0])} onChange={(e) => setVal(e.target.value)}>{prm.options.map((o) => <option key={o}>{o}</option>)}</select></span>;
  else if (prm.type === "text") ctl = <span className="ctl"><input type="text" placeholder={prm.ph} value={value && value !== "none" ? value : ""} onChange={(e) => setVal(e.target.value)} /></span>;
  else if (prm.type === "wb") ctl = <WbControl value={value || {}} setVal={setVal} />;
  else {
    const v = typeof value === "number" ? value : (prm.min < 0 ? 0 : prm.min);
    ctl = <span className="ctl"><input type="range" min={prm.min} max={prm.max} step={prm.step} value={v} onChange={(e) => setVal(+e.target.value)} /><output>{(+v).toFixed(prm.step >= 1 ? 0 : 2)}</output></span>;
  }
  return (
    <div className={"prow" + (on ? "" : " off")}>
      <input type="checkbox" checked={on} onChange={(e) => toggle(e.target.checked)} title="apply this parameter" />
      <span className="plabel">{prm.label}{nowLabel && <span className="muted" style={{ fontSize: 11 }}> · {nowLabel}</span>}</span>
      {ctl}
    </div>
  );
}

function WbControl({ value, setVal }) {
  const mode = value.mode || "camera";
  return (
    <span className="ctl">
      <select value={mode} onChange={(e) => setVal({ ...value, mode: e.target.value })}>
        <option value="camera">camera</option><option value="kelvin">kelvin…</option>
      </select>
      {mode === "kelvin" && <input type="number" min="2000" max="50000" step="50" value={value.temp || 5500}
        onChange={(e) => setVal({ ...value, mode: "kelvin", temp: +e.target.value })} style={{ width: 84 }} />}
    </span>
  );
}

function Review({ ids, overrides, opts, setOpts }) {
  const entries = [];
  for (const [k, v] of Object.entries(overrides.retouch || {})) {
    if (k === "clothing") for (const [ck, cv] of Object.entries(v)) entries.push([`clothing.${ck}`, cv]);
    else if (k === "background") for (const [bk, bv] of Object.entries(v)) entries.push([`background.${bk}`, bv]);
    else entries.push([k, v]);
  }
  for (const [k, v] of Object.entries(overrides.develop || {})) entries.push([`develop.${k}`, v]);
  if (overrides.white_balance) entries.push(["white_balance", overrides.white_balance.mode === "kelvin" ? `${overrides.white_balance.temp}K` : "camera"]);
  return (
    <>
      <div className="hint">Only enabled parameters are overridden — everything else keeps each photo's current values.</div>
      <div className="sect">Photos ({ids.length})</div>
      <div>{fmtIds(ids)}</div>
      <div className="sect">Overrides ({entries.length})</div>
      {entries.length ? entries.map(([k, v]) => <div key={k}>• <b>{k}</b> → {String(fmtV(v))}</div>)
        : <div className="muted" style={{ color: "var(--warn)" }}>None — re-renders with current params.</div>}
      <div className="sect" style={{ marginTop: 14 }}>Options</div>
      <label style={{ display: "block", margin: "6px 0" }}>
        <input type="checkbox" checked={opts.from_raw} onChange={(e) => setOpts({ ...opts, from_raw: e.target.checked })} /> re-develop from RAW
      </label>
      {!SHARE && (
        <>
          <label style={{ display: "block", margin: "6px 0" }}>
            <input type="checkbox" checked={opts.reanalyze} onChange={(e) => setOpts({ ...opts, reanalyze: e.target.checked })} /> re-run AI analysis (uses API)
          </label>
          <div style={{ opacity: opts.reanalyze ? 1 : 0.45, marginLeft: 18 }}>
            <label style={{ marginRight: 14 }}><input type="checkbox" disabled={!opts.reanalyze} checked={opts.subject_only} onChange={(e) => setOpts({ ...opts, subject_only: e.target.checked })} /> subject only</label>
            <label><input type="checkbox" disabled={!opts.reanalyze} checked={opts.skin_exposure} onChange={(e) => setOpts({ ...opts, skin_exposure: e.target.checked })} /> meter on skin</label>
          </div>
        </>
      )}
    </>
  );
}

function fmtV(v) {
  if (typeof v === "number") return +v.toFixed(3);
  if (typeof v === "boolean") return v ? "on" : "off";
  if (v && typeof v === "object") return JSON.stringify(v);
  return v ?? "—";
}
