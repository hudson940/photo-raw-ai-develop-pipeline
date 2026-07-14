import React, { useState } from "react";
import { api } from "../api.js";

export default function Login() {
  const [username, setU] = useState("");
  const [password, setP] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setErr(""); setBusy(true);
    try {
      await api.post("/auth/login", { username, password });
      location.reload();
    } catch (ex) { setErr(ex.message); setBusy(false); }
  };

  return (
    <div className="login">
      <form className="box" onSubmit={submit}>
        <h1>Photo<span>RAW</span></h1>
        <p>Sign in to continue</p>
        <label>Username</label>
        <input value={username} onChange={(e) => setU(e.target.value)} autoFocus autoComplete="username" />
        <label>Password</label>
        <input type="password" value={password} onChange={(e) => setP(e.target.value)} autoComplete="current-password" />
        <button className="primary" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
        {err && <div className="err" style={{ marginTop: 12 }}>{err}</div>}
      </form>
    </div>
  );
}
