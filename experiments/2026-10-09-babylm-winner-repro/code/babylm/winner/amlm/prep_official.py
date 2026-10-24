"""Build AMLM's single-file inputs from the official BabyLM 2025 strict-small text (OSF train_10M.zip, dev.zip).
usage: python prep_official.py TRAIN_DIR DEV_DIR OUT_DIR
Writes, with sha256 + whitespace-word counts in OUT_DIR/manifest.json:
  train_10M.txt        the six *.train files concatenated in sorted name order, blank lines dropped
  dev_small.txt        every k-th non-blank DEV line (k=100, ~104k words): the in-training eval set, same size class
                       as the authors' bb25_small.dev
  train_sample_1M.txt  seed-0 random sample of train lines up to 1M words, in file order (per-token loss probe)
DEV itself stays the official dev split; the official test split is never read."""
import glob, hashlib, json, os, random, sys
tr, dv, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
def lines(d, ext):
    fs = sorted(glob.glob(os.path.join(d, f"*.{ext}"))); assert len(fs) == 6, fs
    return fs, [l.rstrip("\n") for f in fs for l in open(f, encoding="utf-8") if l.strip()]
tf, train = lines(tr, "train"); df, dev = lines(dv, "dev")
rng = random.Random(0); idx = list(range(len(train))); rng.shuffle(idx)
pick, n = [], 0
for i in idx:
    if n >= 1_000_000: break
    pick.append(i); n += len(train[i].split())
files = {"train_10M.txt": train, "dev_small.txt": dev[::100], "train_sample_1M.txt": [train[i] for i in sorted(pick)]}
man = {"train_sources": [os.path.basename(f) for f in tf], "dev_sources": [os.path.basename(f) for f in df]}
for name, ls in files.items():
    p = os.path.join(out, name); open(p, "w", encoding="utf-8").write("\n".join(ls) + "\n")
    man[name] = {"lines": len(ls), "words": sum(len(l.split()) for l in ls), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1); print(json.dumps(man, indent=1))
