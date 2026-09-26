import React, { useState } from "react";
import { api, SHARE } from "../api.js";

// The in-app password screen for a customer album link (replaces the browser's native
// HTTP Basic popup). Themed via the surrounding body theme class.
export default function ShareLogin() {
  const [password, setPw] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setErr(""); setBusy(true);
    try {
      await api.shareLogin(password);
      location.reload();               // cookie set -> config.authed becomes true
    } catch (ex) { setErr(ex.message || "incorrect password"); setBusy(false); }
  };

  return (
    <div className="login share-login">
      <form className="box" onSubmit={submit}>
        <img className="login-logo" src="/logo.png" alt="" />
        <h1 className="album-title">{SHARE?.album || "Album"}</h1>
        <p>Enter the password from your invitation to view your photos.</p>
        <label>Password</label>
        <input type="password" value={password} onChange={(e) => setPw(e.target.value)}
          autoFocus autoComplete="current-password" />
        <button className="primary" disabled={busy}>{busy ? "Opening…" : "View album"}</button>
        {err && <div className="err" style={{ marginTop: 12 }}>{err}</div>}
      </form>
    </div>
  );
}
