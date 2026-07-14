import React from "react";
import { useApp } from "../ctx.js";
import { api } from "../api.js";

export default function Jobs() {
  const app = useApp();
  const cancel = async (id) => { await api.post(`/api/jobs/${id}/cancel`, {}); app.refreshJobs(); };

  return (
    <div className="jobs">
      <div className="jhead"><b>Render jobs</b><span className="spacer" /><button className="ghost" onClick={() => app.setJobsOpen(false)}>✕</button></div>
      {!app.jobs.length && <div className="job muted">No jobs yet.</div>}
      {app.jobs.map((job) => {
        const items = Object.entries(job.items);
        const done = items.filter(([, i]) => i.status === "done").length;
        const err = items.filter(([, i]) => i.status === "error").length;
        const closed = items.filter(([, i]) => ["done", "error", "cancelled"].includes(i.status)).length;
        const pct = Math.round((100 * closed) / items.length);
        const cancellable = job.state === "queued" || job.state === "running";
        return (
          <div className="job" key={job.id}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}>
              <b>{job.id}</b>
              <span className={"badge " + (job.state === "done" ? "done" : job.state === "running" ? "analyzed" : "")}>{job.state}</span>
              <span className="muted">{done}/{items.length} done{err ? `, ${err} failed` : ""}</span>
              <span className="spacer" />
              {cancellable && <button className="ghost" onClick={() => cancel(job.id)}>cancel</button>}
            </div>
            <div className="bar2"><i style={{ width: pct + "%" }} /></div>
            <div>{items.map(([pid, i]) => <span key={pid} className={"pill " + i.status} title={i.error || i.output || ""}>#{pid} {i.status}</span>)}</div>
          </div>
        );
      })}
    </div>
  );
}
