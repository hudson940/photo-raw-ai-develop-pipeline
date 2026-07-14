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

export const api = {
  get: (p) => req("GET", p),
  post: (p, body) => req("POST", p, body),
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

export const imgUrl = (id, { preview = false, base = false, v } = {}) => {
  const q = new URLSearchParams();
  if (v != null) q.set("t", v);
  if (base) q.set("src", "base");        // full-frame image in developed orientation (crop editor)
  else if (preview) q.set("src", "preview");
  const s = q.toString();
  return `${API}/img/${id}${s ? "?" + s : ""}`;
};
export const thumbUrl = (id, v) => `${API}/thumb/${id}?v=${v || 0}`;
