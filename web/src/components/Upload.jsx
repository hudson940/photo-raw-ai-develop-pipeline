import React, { useRef, useState } from "react";
import { useApp } from "../ctx.js";
import { api } from "../api.js";

const RAW_EXT = /\.(cr2|cr3|nef|nrw|arw|srf|sr2|dng|raf|orf|rw2|pef|srw|3fr|erf|kdc|mos|mrw|x3f|iiq)$/i;

export default function Upload() {
  const app = useApp();
  const [items, setItems] = useState([]);   // {name, pct, status, error}
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);

  const add = (fileList) => {
    const files = [...fileList];
    const rows = files.map((f) => ({ file: f, name: f.name, pct: 0, status: RAW_EXT.test(f.name) ? "queued" : "skipped", error: RAW_EXT.test(f.name) ? "" : "not a RAW file" }));
    setItems((s) => [...s, ...rows]);
    upload(rows);
  };

  const upload = async (rows) => {
    setBusy(true);
    for (const row of rows) {
      if (row.status === "skipped") continue;
      const set = (patch) => setItems((s) => s.map((it) => (it === row || it.name === row.name && it.file === row.file ? { ...it, ...patch } : it)));
      set({ status: "uploading" });
      try {
        await api.upload(row.file, (frac) => set({ pct: Math.round(frac * 100) }));
        set({ status: "done", pct: 100 });
      } catch (e) { set({ status: "error", error: e.message }); }
    }
    setBusy(false);
    app.loadPhotos();
  };

  const done = items.filter((i) => i.status === "done").length;

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && !busy && app.setModal(null)}>
      <div className="modal" style={{ width: "min(560px,96vw)" }}>
        <div className="head"><h2>Upload photos</h2><span className="spacer" /><button className="ghost" disabled={busy} onClick={() => app.setModal(null)}>✕</button></div>
        <div className="body">
          <div className={"drop" + (over ? " over" : "")}
            onClick={() => inputRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}>
            <div style={{ fontSize: 26 }}>⬆</div>
            <div>Drop RAW files here, or click to choose</div>
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              CR3 · NEF · ARW · DNG · RAF … They enter the pipeline immediately
              {app.me && app.authEnabled && !app.isSuperAdmin ? ` as your photos` : ""}.
            </div>
          </div>
          <input ref={inputRef} type="file" multiple accept=".cr2,.cr3,.nef,.nrw,.arw,.dng,.raf,.orf,.rw2,.pef,.srw,.3fr,.erf,.kdc,.mos,.mrw,.x3f,.iiq"
            style={{ display: "none" }} onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
          {items.length > 0 && (
            <div style={{ marginTop: 14 }}>
              {items.map((it, idx) => (
                <div className="uprow" key={idx}>
                  <span style={{ flex: "0 0 40%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{it.name}</span>
                  <div className="bar2"><i style={{ width: it.pct + "%", background: it.status === "error" ? "var(--err)" : "var(--accent)" }} /></div>
                  <span className={it.status === "error" ? "err" : "muted"} style={{ flex: "0 0 90px", textAlign: "right", fontSize: 12 }}>
                    {it.status === "error" ? "failed" : it.status === "skipped" ? "skipped" : it.status === "done" ? "queued ✓" : it.pct + "%"}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="foot">
          <span className="muted" style={{ fontSize: 12.5 }}>{done ? `${done} uploaded → queued for processing` : ""}</span>
          <span className="spacer" />
          <button onClick={() => app.setModal(null)} disabled={busy}>Done</button>
        </div>
      </div>
    </div>
  );
}
