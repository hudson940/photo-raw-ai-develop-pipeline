import React, { useEffect, useRef, useState } from "react";
import Cropper from "cropperjs";
import "cropperjs/dist/cropper.css";
import { useApp } from "../ctx.js";
import { saveOutput, imgUrl } from "../api.js";

const RATIOS = [["Free", NaN], ["1:1", 1], ["4:5", 4 / 5], ["5:7", 5 / 7],
  ["3:2", 3 / 2], ["2:3", 2 / 3], ["16:9", 16 / 9]];

// Crop / rotate / straighten in the browser (cropperjs) and save the result straight
// to the photo's output — no backend re-develop. Operates on the current published
// image at full resolution.
export default function CropEditor({ photo, onClose }) {
  const app = useApp();
  const imgRef = useRef(null);
  const cropper = useRef(null);
  const [orient, setOrient] = useState(0);          // 0 / 90 / 180 / 270 (90° steps)
  const [straighten, setStraighten] = useState(0);  // -45..45 fine angle
  const [ratio, setRatio] = useState("Free");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const c = new Cropper(imgRef.current, {
      viewMode: 1, autoCropArea: 1, background: true, responsive: true,
      checkOrientation: false, zoomable: true, movable: true,
    });
    cropper.current = c;
    return () => c.destroy();
  }, []);

  // apply the combined rotation whenever orientation or straighten changes
  useEffect(() => { cropper.current?.rotateTo(orient + straighten); }, [orient, straighten]);

  const changeRatio = (label) => {
    setRatio(label);
    cropper.current?.setAspectRatio(RATIOS.find(([l]) => l === label)[1]);
  };
  const reset = () => { setOrient(0); setStraighten(0); changeRatio("Free"); cropper.current?.reset(); };

  const apply = async () => {
    setBusy(true);
    try {
      const canvas = cropper.current.getCroppedCanvas({
        maxWidth: 8000, maxHeight: 8000, imageSmoothingQuality: "high", fillColor: "#fff",
      });
      const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.92));
      if (!blob) throw new Error("could not render the crop");
      const r = await saveOutput(photo.id, blob);
      app.patchPhoto(photo.id, { output_mtime: r.output_mtime, has_output: true });
      onClose();
    } catch (e) { alert("Crop failed: " + e.message); }
    finally { setBusy(false); }
  };

  return (
    <div className="crop-overlay">
      <div className="subbar">
        <span className="muted">Crop &amp; rotate — saves directly, no re-render.</span>
        <button onClick={() => setOrient((o) => (o + 270) % 360)} title="rotate left 90°">⟲ 90°</button>
        <button onClick={() => setOrient((o) => (o + 90) % 360)} title="rotate right 90°">⟳ 90°</button>
        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          straighten
          <input type="range" min="-45" max="45" step="0.5" value={straighten}
            onChange={(e) => setStraighten(+e.target.value)} style={{ width: 120 }} />
          <span style={{ minWidth: 34, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{straighten}°</span>
        </label>
        <label>ratio&nbsp;
          <select value={ratio} onChange={(e) => changeRatio(e.target.value)}>
            {RATIOS.map(([l]) => <option key={l} value={l}>{l}</option>)}
          </select>
        </label>
        <span className="spacer" />
        <button onClick={reset}>Reset</button>
        <button onClick={onClose}>Cancel</button>
        <button className="primary" disabled={busy} onClick={apply}>{busy ? "Saving…" : "Apply & save"}</button>
      </div>
      <div className="crop-stage">
        <img ref={imgRef} alt="" src={imgUrl(photo.id, { v: photo.output_mtime })} />
      </div>
    </div>
  );
}
