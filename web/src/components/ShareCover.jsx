import React from "react";
import { SHARE } from "../api.js";

// Decorative cover banner shown at the top of a themed customer album (e.g. the
// "quince años" theme). Kept purely presentational — the gallery renders below it.
export default function ShareCover() {
  const name = SHARE?.album || "Album";
  return (
    <section className="cover">
      <div className="cover-particles" aria-hidden="true">
        {Array.from({ length: 18 }).map((_, i) => <span key={i} className="particle" />)}
      </div>
      <div className="cover-inner">
        <div className="cover-eyebrow">Mis</div>
        <h2 className="cover-title">Quince Años</h2>
        <div className="cover-name">{name}</div>
        <div className="cover-rule"><span>✦</span></div>
        <p className="cover-tagline">Selecciona tus fotos favoritas · toca una foto para verla en grande</p>
      </div>
    </section>
  );
}
