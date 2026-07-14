import { createContext, useContext } from "react";

export const AppCtx = createContext(null);
export const useApp = () => useContext(AppCtx);

// compress [1,2,3,7] -> "1-3,7"
export function fmtIds(ids) {
  if (!ids.length) return "";
  const s = [...ids].sort((a, b) => a - b);
  const out = [];
  let lo = s[0], prev = s[0];
  for (const id of s.slice(1)) {
    if (id === prev + 1) { prev = id; continue; }
    out.push(lo === prev ? `${lo}` : `${lo}-${prev}`);
    lo = prev = id;
  }
  out.push(lo === prev ? `${lo}` : `${lo}-${prev}`);
  return out.join(",");
}
