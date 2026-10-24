"""Parse a recipe training log into a loss-curve CSV + PNG (Atila reads every curve).

AMLM log lines: "Total steps: N", "Epoch e, Loss: l, LR: r" (train loss of the last minibatch, adaptive mask),
"----- Eval accuracy: a, Loss: l -----" (fixed 15% MLM on the authors' dev), and with --all_checkpoints
each eval is followed by "Saved checkpoint to: .../checkpoint-STEP", which gives the eval its step.
usage: python curves.py LOG OUT_PREFIX [--epochs 10] [--title T]
"""
import argparse, csv, re

ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("out"); ap.add_argument("--epochs", type=float, default=None)
ap.add_argument("--title", default="")
a = ap.parse_args()

total, rows, pending_eval, last_step = None, [], None, 0
for line in open(a.log, errors="replace"):
    if m := re.search(r"Total steps: (\d+)", line): total = int(m[1])
    elif m := re.search(r"Epoch ([\d.]+), Loss: ([\d.naninf]+), LR: ([\d.e+-]+)", line):
        ep = float(m[1]); epochs = a.epochs or 10.0
        last_step = round(ep * total / epochs) if total else len(rows)
        rows.append(dict(step=last_step, epoch=ep, split="train", loss=m[2], acc="", lr=m[3]))
    elif m := re.search(r"Eval accuracy: ([\d.]+), Loss: ([\d.naninf]+)", line):
        if pending_eval:  # previous eval had no checkpoint line: attach it to the last train step
            rows.append(dict(step=last_step, epoch="", split="eval", loss=pending_eval[1], acc=pending_eval[0], lr=""))
        pending_eval = (m[1], m[2])
    elif (m := re.search(r"checkpoint-(\d+)", line)) and pending_eval:
        s = int(m[1]); rows.append(dict(step=s, epoch=s * (a.epochs or 10.0) / total if total else "", split="eval",
                                        loss=pending_eval[1], acc=pending_eval[0], lr="")); pending_eval = None
    elif m := re.search(r"Final eval accuracy: ([\d.]+), Loss: ([\d.naninf]+)", line):
        rows.append(dict(step=total or last_step, epoch=a.epochs or 10.0, split="eval_final", loss=m[2], acc=m[1], lr=""))
if pending_eval:  # eval without a following checkpoint line: attach to the last train step
    rows.append(dict(step=last_step, epoch="", split="eval", loss=pending_eval[1], acc=pending_eval[0], lr=""))

with open(a.out + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step", "epoch", "split", "loss", "acc", "lr"]); w.writeheader(); w.writerows(rows)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.5))
for split, style in (("train", "-"), ("eval", "o-"), ("eval_final", "s")):
    pts = [(r["step"], float(r["loss"])) for r in rows if r["split"] == split]
    if pts: ax.plot(*zip(*pts), style, label=split, markersize=4)
ax.set_xlabel("optimizer step"); ax.set_ylabel("MLM loss (nats/token)"); ax.set_title(a.title or a.log)
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(a.out + ".png", dpi=120)
print(f"wrote {a.out}.csv ({len(rows)} rows) and {a.out}.png")
