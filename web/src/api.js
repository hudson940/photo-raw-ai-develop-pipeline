// Runtime config injected by the backend as window.__CONFIG__ (and window.SHARE for
// customer share links). In share mode every request is prefixed with /share/<token>.
const CONFIG = window.__CONFIG__ || {};
export const SHARE = CONFIG.share || window.SHARE || null;
export const API = SHARE ? SHARE.prefix : "";

async function req(method, path, body, opts = {}) {
  const init = { method, headers: {}, ...opts };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(API + path, init);
  const ctype = res.headers.get("Content-Type") || "";
  const data = ctype.includes("application/json") ? await res.json().catch(() => ({})) : null;
  if (!res.ok) {
    const msg = (data && data.error) || `${res.status} ${res.statusText}`;
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

// Album (share) presentation theme + whether the visitor has entered the password yet.
export const SHARE_THEME = SHARE ? (SHARE.theme || "default") : null;

export const api = {
  get: (p) => req("GET", p),
  post: (p, body) => req("POST", p, body),
  // customer album login: exchanges the link password for a session cookie
  shareLogin: (password) => req("POST", "/login", { password }),
  // raw upload: body is the File; filename passed as a query param
  upload: (file, onProgress) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API}/api/upload?filename=${encodeURIComponent(file.name)}`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch {}
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(new Error(data.error || `upload failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error("network error during upload"));
      xhr.send(file);
    }),
};

// Save a browser-edited image (crop/rotate) as the photo's output — raw JPEG body,
// no backend render. Returns { output_mtime } for cache-busting.
export async function saveOutput(id, blob) {
  const res = await fetch(`${API}/api/photos/${id}/output`, {
    method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `save failed (${res.status})`);
  return data;
}

// Convert any browser-decodable image file to a JPEG blob. Used to replace a photo's
// edited version with an externally-edited image while keeping the same name.
export function fileToJpegBlob(file, quality = 0.92) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff";                       // flatten any transparency onto white
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("could not encode image"))),
        "image/jpeg", quality);
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("unsupported image file")); };
    img.src = url;
  });
}

export const imgUrl = (id, { preview = false, base = false, original = false, v } = {}) => {
  const q = new URLSearchParams();
  if (v != null) q.set("t", v);
  if (base) q.set("src", "base");        // full-frame image in developed orientation (crop editor)
  else if (original) q.set("src", "original");  // full-res render of the untouched RAW
  else if (preview) q.set("src", "preview");
  const s = q.toString();
  return `${API}/img/${id}${s ? "?" + s : ""}`;
};
export const thumbUrl = (id, v) => `${API}/thumb/${id}?v=${v || 0}`;

// Download URLs (the server sends Content-Disposition: attachment). One photo, or a
// zip of many. Honors each photo's developed-vs-original delivery choice.
// An optional `preset` selects a size/quality tier (high|print|medium|small); "high"
// (default) is the full-res stored file, so we omit the param to keep those URLs clean.
export const downloadUrl = (id, preset) =>
  `${API}/api/photos/${id}/download` + (preset && preset !== "high" ? `?preset=${preset}` : "");
export const zipUrl = (ids, preset) =>
  `${API}/api/download?ids=${ids.join(",")}` + (preset && preset !== "high" ? `&preset=${preset}` : "");

// Trigger a browser download for a same-origin attachment URL.
export function triggerDownload(url) {
  const a = document.createElement("a");
  a.href = url;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}
