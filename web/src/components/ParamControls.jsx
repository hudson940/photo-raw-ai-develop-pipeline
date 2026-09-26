import React from "react";
import { SHARE } from "../api.js";
import { fmtV } from "../redoParams.js";

// One redo parameter: an "apply" checkbox + its control. Only checked rows are sent.
export function ParamRow({ prm, on, value, toggle, setVal, showNow }) {
  let ctl;
  if (prm.type === "bool") ctl = <label className="ctl"><input type="checkbox" checked={value === true} onChange={(e) => setVal(e.target.checked)} /> on</label>;
  else if (prm.type === "select") ctl = <span className="ctl"><select value={String(value ?? prm.options[0])} onChange={(e) => setVal(e.target.value)}>{prm.options.map((o) => <option key={o}>{o}</option>)}</select></span>;
  else if (prm.type === "text") ctl = <span className="ctl"><input type="text" placeholder={prm.ph} value={value && value !== "none" ? value : ""} onChange={(e) => setVal(e.target.value)} /></span>;
  else if (prm.type === "wb") ctl = <WbControl value={value || {}} setVal={setVal} />;
  else {
    const v = typeof value === "number" ? value : (prm.min < 0 ? 0 : prm.min);
    ctl = (
      <span className="ctl">
        <input type="range" min={prm.min} max={prm.max} step={prm.step} value={v} onChange={(e) => setVal(+e.target.value)}
          onDoubleClick={() => toggle(false)} title="double-click to stop overriding" />
        <output>{(+v).toFixed(prm.step >= 1 ? 0 : 2)}</output>
      </span>
    );
  }
  return (
    <div className={"prow" + (on ? "" : " off") + (prm.type === "wb" ? " wide" : "")}>
      <input type="checkbox" checked={on} onChange={(e) => toggle(e.target.checked)} title="apply this parameter" />
      <span className="plabel">{prm.label}{showNow && <span className="now"> · now {String(fmtV(value))}</span>}</span>
      {ctl}
    </div>
  );
}

export function WbControl({ value, setVal }) {
  const mode = value.mode || "camera";
  return (
    <span className="ctl wb">
      <select value={mode} onChange={(e) => setVal({ ...value, mode: e.target.value })}>
        <option value="camera">camera</option><option value="kelvin">kelvin…</option>
      </select>
      {mode === "kelvin" && <>
        <input type="number" min="2000" max="50000" step="50" value={value.temp || 5500} title="temperature (K)"
          onChange={(e) => setVal({ ...value, mode: "kelvin", temp: +e.target.value })} style={{ width: 80 }} />
        <input type="number" min="-50" max="50" step="1" value={value.tint || 0} title="tint (−green … +magenta)"
          onChange={(e) => setVal({ ...value, mode: "kelvin", tint: +e.target.value })} style={{ width: 58 }} />
      </>}
    </span>
  );
}

// Render options that are not parameters: re-develop from the RAW, re-run the AI.
export function RedoOptions({ opts, setOpts }) {
  return (
    <div className="redo-opts">
      <label><input type="checkbox" checked={opts.from_raw} onChange={(e) => setOpts({ ...opts, from_raw: e.target.checked })} /> re-develop from RAW</label>
      {!SHARE && <>
        <label><input type="checkbox" checked={opts.reanalyze} onChange={(e) => setOpts({ ...opts, reanalyze: e.target.checked })} /> re-run AI analysis (uses API)</label>
        <div className="sub" style={{ opacity: opts.reanalyze ? 1 : 0.45 }}>
          <label><input type="checkbox" disabled={!opts.reanalyze} checked={opts.subject_only} onChange={(e) => setOpts({ ...opts, subject_only: e.target.checked })} /> subject only</label>
          <label><input type="checkbox" disabled={!opts.reanalyze} checked={opts.skin_exposure} onChange={(e) => setOpts({ ...opts, skin_exposure: e.target.checked })} /> meter on skin</label>
        </div>
      </>}
    </div>
  );
}

export const DEFAULT_OPTS = { from_raw: false, reanalyze: false, subject_only: false, skin_exposure: false };

export const redoBody = (ids, overrides, opts) => ({
  ids, overrides, from_raw: opts.from_raw, reanalyze: opts.reanalyze,
  subject_only: opts.reanalyze && opts.subject_only, skin_exposure: opts.reanalyze && opts.skin_exposure,
});
