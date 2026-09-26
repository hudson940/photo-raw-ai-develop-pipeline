// Every redo option the backend accepts (pipeline/webui.py normalize_overrides mirrors the
// `python -m pipeline.redo` flags). Shared by the batch Redo wizard and the photo editor.

export const SKIN_TYPES = ["auto", "fair", "light", "medium", "olive", "tan", "brown", "deep"];
export const BG_COLORS = ["gray", "white", "black", "charcoal", "blue", "navy", "teal", "green", "red", "maroon", "pink", "purple", "beige", "brown"];
export const BG_ACTIONS = ["auto", "keep", "clean", "expand", "smooth", "blur", "studio", "replace"];

export const SECTIONS = [
  { id: "light", title: "Light", icon: "☀", hint: "Exposure / white balance / shadows / highlights re-develop the RAW.", params: [
    { g: "wb", k: "wb", label: "White balance", type: "wb" },
    { g: "develop", k: "exposure_ev", label: "Exposure (EV)", min: -2, max: 2, step: 0.05 },
    { g: "develop", k: "shadows", label: "Shadows", min: -100, max: 100, step: 1 },
    { g: "develop", k: "highlights", label: "Highlights", min: -100, max: 100, step: 1 },
    { g: "retouch", k: "subject_exposure", label: "Subject exposure (EV)", min: -2, max: 2, step: 0.1 },
    { g: "retouch", k: "background_exposure", label: "Background exposure (EV)", min: -2, max: 2, step: 0.1 },
    { g: "retouch", k: "intensity", label: "Retouch intensity", type: "select", options: ["subtle", "natural", "polished"] },
  ] },
  { id: "auto", title: "Auto levels", icon: "◐", hint: "Meters face skin / subject / background toward a midtone target and corrects each through its mask. 0 turns a layer off.", params: [
    { g: "retouch", k: "auto_levels", label: "Enable auto levels", type: "bool" },
    { g: "retouch", k: "auto_skin_target", label: "Skin target", min: 0, max: 1, step: 0.01 },
    { g: "retouch", k: "auto_subject_target", label: "Subject target", min: 0, max: 1, step: 0.01 },
    { g: "retouch", k: "auto_background_target", label: "Background target", min: 0, max: 1, step: 0.01 },
  ] },
  { id: "skin", title: "Skin", icon: "◉", hint: "Tone correction targets natural colour per skin type. Warmth / richness / red ride on tone correction.", params: [
    { g: "retouch", k: "skin_smoothing", label: "Skin smoothing", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "remove_blemishes", label: "Remove blemishes", type: "bool" },
    { g: "retouch", k: "skin_tone_correction", label: "Tone correction", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "skin_type", label: "Skin type", type: "select", options: SKIN_TYPES },
    { g: "retouch", k: "skin_warmth", label: "Warmth (yellow↔orange)", min: -1, max: 1, step: 0.05 },
    { g: "retouch", k: "skin_saturation", label: "Richness", min: 0.5, max: 1.8, step: 0.05 },
    { g: "retouch", k: "skin_luminance", label: "Skin luminance", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "skin_red", label: "Red flush (cheeks)", min: 0, max: 1, step: 0.05 },
  ] },
  { id: "face", title: "Face", icon: "☺", params: [
    { g: "retouch", k: "brighten_eyes", label: "Brighten eyes", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "iris_enhance", label: "Iris pop", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "whiten_teeth", label: "Whiten teeth", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "lip_enhance", label: "Lip enhancement", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "reduce_dark_circles", label: "Dark circles", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "reduce_dewlap", label: "Double chin", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "tame_highlights", label: "Tame shiny highlights", min: 0, max: 1, step: 0.05 },
  ] },
  { id: "hair", title: "Hair", icon: "≋", params: [
    { g: "retouch", k: "hair_texture", label: "Hair texture", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "hair_shimmer", label: "Hair shimmer", min: 0, max: 1, step: 0.05 },
    { g: "retouch", k: "hair_defrizz", label: "Defrizz", min: 0, max: 1, step: 0.05 },
  ] },
  { id: "clothing", title: "Clothing", icon: "◇", params: [
    { g: "retouch", k: "clothing_contrast", label: "Clothing contrast", min: 0, max: 1, step: 0.05 },
    { g: "clothing", k: "color", label: "Cloth colour", type: "text", ph: "red, navy…" },
    { g: "clothing", k: "color_pop", label: "Colour pop", min: 0, max: 1, step: 0.05 },
    { g: "clothing", k: "luminance", label: "Luminance", min: -1, max: 1, step: 0.05 },
    { g: "clothing", k: "shadows", label: "Shadows", min: -1, max: 1, step: 0.05 },
    { g: "clothing", k: "blacks", label: "Blacks", min: -1, max: 1, step: 0.05 },
    { g: "clothing", k: "whites", label: "Whites", min: -1, max: 1, step: 0.05 },
  ] },
  { id: "background", title: "Background", icon: "▦", hint: "auto cleans a studio backdrop (real scenes kept) · clean = only the backdrop cloth, by colour · expand = also cover walls/stands · backdrop colour re-tints the cleaned cloth (e.g. gray → white) · studio colour is for the studio action · floor turns silver reflector panels into water or foil.", params: [
    { g: "background", k: "action", label: "Action", type: "select", options: BG_ACTIONS },
    { g: "background", k: "floor", label: "Floor / reflector", type: "select", options: ["none", "water", "foil"] },
    { g: "background", k: "backdrop_color", label: "Backdrop colour (clean / expand)", type: "select", options: ["auto", ...BG_COLORS], show: (v) => ["auto", "clean", "expand"].includes(v["background.action"] || "auto") },
    { g: "background", k: "color", label: "Studio colour", type: "select", options: BG_COLORS, show: (v) => v["background.action"] === "studio" },
    { g: "background", k: "replace_prompt", label: "Replace prompt", type: "text", ph: "soft window light…", show: (v) => v["background.action"] === "replace" },
  ] },
  { id: "erase", title: "Erase", icon: "⌫", hint: "Paint the mask with the Erase tool; here you change how the saved mask is filled, or drop it.", params: [
    { g: "erase", k: "method", label: "Fill method", type: "select", options: ["content-aware", "generative"] },
    { g: "erase", k: "prompt", label: "Generative prompt", type: "text", ph: "what to paint instead" },
    { g: "erase", k: "protect_subject", label: "Protect subject", type: "bool" },
    { g: "erase", k: "clear", label: "Remove saved mask", type: "bool" },
  ] },
];

export const pkey = (prm) => `${prm.g}.${prm.k}`;

// current value of a parameter for a photo (detail) or the defaults
export function currentValue(prm, detail, defs) {
  const r = detail ? detail.retouch || {} : defs || {};
  switch (prm.g) {
    case "retouch": return r[prm.k];
    case "clothing": return (r.clothing || {})[prm.k];
    case "background": {
      const v = (r.background || {})[prm.k];
      if (prm.k === "floor") return v || "none";
      if (prm.k === "backdrop_color") return v || "auto";
      return v;
    }
    case "erase": return prm.k === "clear" ? false : (r.erase || {})[prm.k];
    case "develop": return detail ? (detail.develop || {})[prm.k] : 0;
    case "wb": return detail ? detail.develop?.white_balance || {} : {};
    default: return undefined;
  }
}

// enabled parameters -> the overrides payload for POST /api/redo
export function buildOverrides(enabled, values) {
  const ov = { retouch: {}, develop: {}, white_balance: {} };
  for (const s of SECTIONS) for (const prm of s.params) {
    const k = pkey(prm);
    if (!enabled[k]) continue;
    const v = values[k];
    if (prm.g === "wb") Object.assign(ov.white_balance, v);
    else if (prm.g === "retouch") ov.retouch[prm.k] = v;
    else if (prm.g === "develop") ov.develop[prm.k] = v;
    else if (prm.g === "erase") {
      const e = (ov.retouch.erase ||= {});
      if (prm.k === "clear") { if (v) e.mask = ""; }
      else e[prm.k] = v;
    } else (ov.retouch[prm.g] ||= {})[prm.k] = v;   // clothing / background
  }
  for (const g of ["retouch", "develop", "white_balance"]) if (!Object.keys(ov[g]).length) delete ov[g];
  if (ov.retouch?.erase && !Object.keys(ov.retouch.erase).length) delete ov.retouch.erase;
  return ov;
}

// flat [label, value] list of an overrides payload, for review screens
export function describeOverrides(ov) {
  const out = [];
  for (const [k, v] of Object.entries(ov.retouch || {})) {
    if (v && typeof v === "object") for (const [ck, cv] of Object.entries(v)) out.push([`${k}.${ck}`, cv]);
    else out.push([k, v]);
  }
  for (const [k, v] of Object.entries(ov.develop || {})) out.push([`develop.${k}`, v]);
  if (ov.white_balance) out.push(["white_balance", ov.white_balance.mode === "kelvin"
    ? `${ov.white_balance.temp || 5500}K${ov.white_balance.tint ? ` tint ${ov.white_balance.tint}` : ""}` : "camera"]);
  return out;
}

export function fmtV(v) {
  if (typeof v === "number") return +v.toFixed(3);
  if (typeof v === "boolean") return v ? "on" : "off";
  if (v === "") return "(cleared)";
  if (v && typeof v === "object") return JSON.stringify(v);
  return v ?? "—";
}
