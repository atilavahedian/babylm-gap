"""How much extra rare-word exposure can late-epoch window reweighting buy? Rare = train_10M normal-form count 1-63.
Windows of W words tiled over each source file (collapsed lines, headers dropped). Reports the share of tokens that are
rare words or within 3 words after one (the focus set), its per-window CV, and the exposure multiplier of sampling
windows with probability proportional to (1 + kappa * r / mean r)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rare_words as RW
RAW = Path(sys.argv[1])
c10 = RW.count_split(RAW, "train_10M", "train")
words = []
for s in RW.SOURCES:
    for ln in (RAW / "train_10M" / f"{s}.train").read_text(encoding="utf-8").split("\n"):
        if RW.is_header(ln):
            continue
        words.extend(ln.split())
n = len(words)
c = np.array([c10.get(RW.norm(w), 0) for w in words])
rare = (c >= 1) & (c <= 63)
never = c == 0
focus = rare.copy()
for d in (1, 2, 3):
    focus[d:] |= rare[:-d]
print(f"train_10M words {n:,}; rare share {rare.mean():.3%}; never-seen-in-train (count 0) {never.mean():.3%}; focus set (rare + 3 following) {focus.mean():.3%}")
for W in (640, 320, 160, 80):
    nw = n // W
    r = rare[: nw * W].reshape(nw, W).sum(1).astype(float)
    f = focus[: nw * W].reshape(nw, W).sum(1).astype(float)
    cv = r.std() / r.mean()
    for kappa in (1.0, 2.0, 4.0):
        w = 1 + kappa * r / r.mean()
        w = w / w.sum()
        mult = (w * r).sum() * nw / r.sum()
        multf = (w * f).sum() * nw / f.sum()
        print(f"W={W:4d} words: rare/window mean {r.mean():6.1f} CV {cv:.2f} | kappa {kappa}: rare exposure x{mult:.2f}, focus-set exposure x{multf:.2f}, other words x{((w * (W - f)).sum() * nw / (W * nw - f.sum())):.2f}")
