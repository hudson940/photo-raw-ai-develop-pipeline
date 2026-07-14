import React, { useEffect, useState } from "react";
import { useApp } from "../ctx.js";
import { api } from "../api.js";

export default function Users() {
  const app = useApp();
  const [users, setUsers] = useState(null);
  const [err, setErr] = useState("");
  const [nu, setNu] = useState({ username: "", password: "", role: "editor" });

  const load = async () => {
    setErr("");
    try { setUsers((await api.get("/api/users")).users || []); }
    catch (e) { setErr(e.message); setUsers([]); }
  };
  useEffect(() => { load(); }, []);

  const create = async () => {
    setErr("");
    try { await api.post("/api/users", nu); setNu({ username: "", password: "", role: "editor" }); load(); }
    catch (e) { setErr(e.message); }
  };
  const setRole = async (id, role) => { await api.post(`/api/users/${id}/role`, { role }); load(); };
  const toggle = async (id, enabled) => { await api.post(`/api/users/${id}/enabled`, { enabled }); load(); };
  const del = async (id) => { if (!confirm("Delete this user?")) return; await api.post(`/api/users/${id}/delete`, {}); load(); };

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && app.setModal(null)}>
      <div className="modal">
        <div className="head"><h2>Users</h2><span className="spacer" /><button className="ghost" onClick={() => app.setModal(null)}>✕</button></div>
        <div className="body">
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "flex-end", marginBottom: 14 }}>
            <div style={{ flex: 1, minWidth: 120 }}><label className="muted" style={{ fontSize: 12 }}>Username</label>
              <input value={nu.username} onChange={(e) => setNu({ ...nu, username: e.target.value })} style={{ width: "100%" }} /></div>
            <div style={{ flex: 1, minWidth: 120 }}><label className="muted" style={{ fontSize: 12 }}>Password</label>
              <input value={nu.password} onChange={(e) => setNu({ ...nu, password: e.target.value })} placeholder="min 6" style={{ width: "100%" }} /></div>
            <div><label className="muted" style={{ fontSize: 12 }}>Role</label>
              <select value={nu.role} onChange={(e) => setNu({ ...nu, role: e.target.value })} style={{ display: "block" }}>
                <option value="editor">editor</option><option value="super_admin">super admin</option></select></div>
            <button className="primary" onClick={create}>Create</button>
          </div>
          {err && <div className="err" style={{ marginBottom: 10 }}>{err}</div>}
          {users === null ? <div className="muted">Loading…</div> : (
            <table className="utable">
              <thead><tr><th>User</th><th>Email</th><th>Role</th><th /></tr></thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id}>
                    <td><b>{u.username}</b>{!u.enabled && <span className="muted"> (disabled)</span>}</td>
                    <td className="muted">{u.email}</td>
                    <td>
                      <select value={u.roles.includes("super_admin") ? "super_admin" : "editor"} onChange={(e) => setRole(u.id, e.target.value)}>
                        <option value="editor">editor</option><option value="super_admin">super admin</option>
                      </select>
                    </td>
                    <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                      <button onClick={() => toggle(u.id, !u.enabled)}>{u.enabled ? "Disable" : "Enable"}</button>{" "}
                      <button onClick={() => del(u.id)}>Delete</button>
                    </td>
                  </tr>
                ))}
                {!users.length && <tr><td colSpan="4" className="muted">No users yet.</td></tr>}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
