"""Queue status: python -m pipeline.status"""

import json

from .config import CONFIG
from . import db

STATE_ORDER = ["pending", "previewed", "analyzed", "developed", "retouched", "done", "review", "failed"]


def main() -> None:
    conn = db.connect(CONFIG.db_path)
    counts = db.counts_by_state(conn)
    total = sum(counts.values())
    print(f"Queue: {total} photo(s)  ({CONFIG.db_path})")
    for state in STATE_ORDER:
        if counts.get(state):
            print(f"  {state:10s} {counts[state]}")

    problems = conn.execute(
        "SELECT id, filename, state, confidence, error FROM photos"
        " WHERE state IN ('failed', 'review') ORDER BY updated_at DESC LIMIT 10"
    ).fetchall()
    if problems:
        print("\nNeeds attention:")
        for r in problems:
            conf = f" conf={r['confidence']:.2f}" if r["confidence"] is not None else ""
            print(f"  #{r['id']} [{r['state']}]{conf} {r['filename']}: {r['error']}")

    latest = conn.execute(
        "SELECT filename, analysis_json FROM photos WHERE analysis_json IS NOT NULL"
        " ORDER BY updated_at DESC LIMIT 1"
    ).fetchone()
    if latest:
        print(f"\nLatest analysis ({latest['filename']}):")
        print(json.dumps(json.loads(latest["analysis_json"]), indent=2))
    conn.close()


if __name__ == "__main__":
    main()
