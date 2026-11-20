"""Run the held-out evaluation for every development-selected gate of one detector stream."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from geometry_gates import OUT_ROOT, run_arm  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: holdout_shard.py <detector>")
    detector = sys.argv[1]
    selection = json.loads((OUT_ROOT / "selection.json").read_text(encoding="utf-8"))
    for family, item in selection["families"].items():
        if item.get("status") != "completed":
            continue
        value = float(item["selected"]["value"])
        run_arm(detector, family, value, "holdout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
