#!/usr/bin/env python
"""Markdown tables for the report from outputs/track1_submission_episodes/summary.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "outputs" / "track1_submission_episodes" / "summary.json").read_text())
rows = [S[k] for k in sorted(S, key=int) if S[k].get("ok")]
print("| ep | object | scored frames (stretches) | MHR fit err mm (all / scored) | repaired % | root depth range m | grasp % scored | own ACC-H | own ACC-O | own PEN cm (before push) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    p = r["proxy"]
    print(f"| {r['episode']} | {r['object']} | {r['n_scored']} ({r['n_stretches']}) | {r['fit']['fit_err_mm_mean']:.1f} / "
          f"{r['fit_err_mm_scored']:.1f} | {100 * r.get('repaired_frac', 0):.1f} | {r['root'].get('depth_range_m', float('nan')):.2f} | "
          f"{100 * r['grasp_frac_scored']:.0f} | {p['acc_h_own_cm']:.3f} | {p['acc_o_own_cm']:.3f} | "
          f"{p['pen_own_cm']:.4f} ({r['pen']['pen_before_cm']:.4f}) |")
w = np.array([r["n_scored"] for r in rows], float)
def m(f):
    v = np.array([f(r) for r in rows], float)
    return v.mean(), (v * w).sum() / w.sum()
for name, f in [("fit err mm (all frames)", lambda r: r["fit"]["fit_err_mm_mean"]),
                ("fit err mm (scored)", lambda r: r["fit_err_mm_scored"]),
                ("own ACC-H cm/f^2", lambda r: r["proxy"]["acc_h_own_cm"]),
                ("own ACC-O cm/f^2", lambda r: r["proxy"]["acc_o_own_cm"]),
                ("own ACC-O before PEN push", lambda r: r["proxy"]["acc_o_own_before_push_cm"]),
                ("own PEN cm", lambda r: r["proxy"]["pen_own_cm"]),
                ("PEN before push cm", lambda r: r["pen"]["pen_before_cm"])]:
    a, b = m(f)
    print(f"- {name}: episode mean {a:.4f}, scored-frame-weighted {b:.4f}")
print(f"- episodes ok: {len(rows)}/{len(S)}, scored frames {int(w.sum())}")
