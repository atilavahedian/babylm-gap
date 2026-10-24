"""Plot a BabyLM loss-curve CSV to PNG. Usage: python plot_curve.py curve.csv [out.png]
Plots every column whose name contains 'loss' against words_seen_est (or step)."""
import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".png"
rows = list(csv.DictReader(open(src)))
x = "words_seen_est" if rows and "words_seen_est" in rows[0] else "step"
fig, ax = plt.subplots(figsize=(8, 5))
for col in rows[0]:
    if "loss" not in col: continue
    pts = [(float(r[x]), float(r[col])) for r in rows if r[col] not in ("", "nan")]
    if pts: ax.plot([p[0] / (1e6 if x == "words_seen_est" else 1) for p in pts], [p[1] for p in pts], label=col, marker="." if "dev" in col else None)
ax.set_xlabel("words seen (M)" if x == "words_seen_est" else x); ax.set_ylabel("nats / token"); ax.grid(alpha=.3); ax.legend()
ax.set_title(src.split("/")[-2] if "/" in src else src)
fig.tight_layout(); fig.savefig(out, dpi=120); print(out)
