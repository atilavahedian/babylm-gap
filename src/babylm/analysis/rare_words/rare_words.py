"""Rare-word anatomy of BabyLM dev vs train_10M (normal-form words, as the instruments count them).

Usage: python3 -I rare_words.py <raw_root> <tokenizer.json> <out_dir>
raw_root has train_10M/, train_100M/, dev/ with the six source files each.
Outputs JSON + CSV summaries under out_dir. Stdlib + tokenizers only.
"""
from __future__ import annotations

import collections
import json
import re
import string
import sys
from pathlib import Path

PUNCT = string.punctuation + "“”‘’«»…—–"
SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
SWB_PIECE = 2000


def norm(word: str) -> str:
    w = word.lower()
    s = w.strip(PUNCT)
    return s if s else w


def is_header(line: str) -> bool:
    return line.startswith("= = =") and line.rstrip().endswith("= = =")


def count_split(root: Path, split: str, ext: str) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for s in SOURCES:
        p = root / split / f"{s}.{ext}"
        with open(p, encoding="utf-8") as f:
            for line in f:
                if is_header(line):
                    continue
                c.update(norm(w) for w in line.split())
    return c


def docs(lines: list[str], source: str):
    """Yield (doc_id, list of lines) following babylm/data/corpus.py: blank-line blocks; switchboard 2000-line pieces."""
    if source == "switchboard":
        for i in range(0, len(lines), SWB_PIECE):
            yield i // SWB_PIECE, lines[i:i + SWB_PIECE]
        return
    cur, did = [], 0
    for ln in lines:
        if ln.strip() == "":
            if cur:
                yield did, cur
                did += 1
                cur = []
        else:
            cur.append(ln)
    if cur:
        yield did, cur


SUFFIXES = ["'s", "s'", "n't", "ing", "ings", "ed", "es", "s", "d", "ly", "er", "ers", "est", "ies", "ied", "ier",
            "iest", "ily", "ness", "nesses", "ment", "ments", "tion", "tions", "sion", "ation", "ations", "able", "ible",
            "ful", "less", "ish", "ous", "al", "ity", "ities", "ize", "ized", "izes", "izing", "ise", "ised", "ises",
            "ising", "ism", "ist", "ists", "ive", "ic", "ical", "ian", "ians", "ship", "hood", "dom", "ward", "wards",
            "like", "y", "en", "ened", "ening", "ess", "ette", "ling", "let", "ry", "ery", "ure", "age", "ance", "ence",
            "ant", "ent", "or", "ors", "ing's", "ism's", "man", "men", "woman", "women"]
PREFIXES = ["un", "re", "dis", "in", "im", "il", "ir", "non", "pre", "over", "under", "mis", "sub", "super", "anti",
            "de", "inter", "trans", "semi", "co", "counter", "out", "up", "fore", "post", "pro", "ex", "mid", "self",
            "half", "auto", "bi", "tri", "multi", "micro", "macro", "neo", "pseudo", "a", "be", "en", "em"]
INFLECT = {"'s", "s'", "n't", "ing", "ings", "ed", "es", "s", "d", "ies", "ied", "er", "ers", "est", "ier", "iest"}


def candidates(w: str):
    """Possible base forms after stripping one suffix (with simple spelling repairs) or one prefix."""
    out = []
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            b = w[: -len(suf)]
            kind = "inflection" if suf in INFLECT else "derivation"
            out.append((b, kind, suf))
            if suf in ("ing", "ed", "er", "est", "ers", "ings", "ening", "ened", "y", "ity", "ation", "ize", "ise", "al", "ous", "ic", "ive", "able", "ible"):
                out.append((b + "e", kind, suf))  # making -> make, famous? (fame -> fam+ous), nervous
            if suf in ("ies", "ied", "ier", "iest", "ily", "ities") and b and not b.endswith("y"):
                out.append((b + "y", kind, suf))  # tries -> try, happily -> happy
            if len(b) >= 4 and b[-1] == b[-2] and b[-1] not in "aeiou":
                out.append((b[:-1], kind, suf))  # running -> run, bigger -> big
    for pre in PREFIXES:
        if w.startswith(pre) and len(w) - len(pre) >= 4:
            out.append((w[len(pre):], "derivation", pre + "-"))
            b = w[len(pre):]
            if b.startswith("-"):
                out.append((b[1:], "derivation", pre + "-"))
    return out


def classify(surface: str, n: str, c10: collections.Counter, c100: collections.Counter, tok):
    """Return (category, detail) for a normal-form word n with train_10M count 0."""
    if any(ch.isdigit() for ch in n):
        return "number/digits", ""
    if "-" in n.strip("-") or "/" in n:
        parts = [p for p in re.split(r"[-/]", n) if p]
        if parts and all(c10.get(p, 0) > 0 for p in parts):
            return "hyphen/slash compound of seen parts", "+".join(parts)
        return "hyphen/slash, part unseen", ""
    if not any(ch.isalpha() for ch in n):
        return "symbols only", ""
    if not re.fullmatch(r"[a-z'’]+", n):
        return "non-ascii letters", ""
    # inflection / derivation (one affix) of a seen word
    best = None
    for b, kind, aff in candidates(n):
        cb = c10.get(b, 0)
        if cb > 0 and (best is None or cb > best[0]):
            best = (cb, kind, aff, b)
    if best is not None:
        return best[1] + " of seen word", f"{best[3]}+{best[2]}" if not best[2].endswith("-") else f"{best[2]}{best[3]}"
    # two-affix (e.g. un+help+ful+ness): strip twice
    for b, kind, aff in candidates(n):
        for b2, kind2, aff2 in candidates(b):
            if c10.get(b2, 0) >= 2:
                return "two-affix derivation of seen word", f"{b2}+{aff2}+{aff}"
    # closed compound of two seen words
    for i in range(3, len(n) - 2):
        a, b = n[:i], n[i:]
        if c10.get(a, 0) >= 10 and c10.get(b, 0) >= 10:
            return "closed compound of seen words", f"{a}+{b}"
    cap = surface.strip(PUNCT)[:1].isupper()
    if cap:
        return "capitalized novel word (name-like)", ""
    return "lowercase novel word", ""


def band(c: int) -> str:
    if c == 0:
        return "0"
    if c <= 63:
        return "1-63"
    if c < 8192:
        return "64-8k"
    return "8k+"


def main() -> None:
    root, tok_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    print("counting train_10M", flush=True)
    c10 = count_split(root, "train_10M", "train")
    print(f"train_10M: {sum(c10.values()):,} words, {len(c10):,} normal-form types", flush=True)
    print("counting train_100M", flush=True)
    c100 = count_split(root, "train_100M", "train")
    print(f"train_100M: {sum(c100.values()):,} words, {len(c100):,} normal-form types", flush=True)

    tot = collections.Counter()           # band -> dev words
    tot_src = collections.Counter()       # (source, band) -> words
    rep = collections.Counter()           # (band, 'first'|'repeat_doc') -> words  (repeat of the normal form earlier in the document)
    rep300 = collections.Counter()        # (band, 'first'|'repeat') within the previous 300 words (chunk-like)
    cat0 = collections.Counter()          # never-seen: category -> words
    cat0_src = collections.Counter()      # (source, category) -> words
    cat0_in100 = collections.Counter()    # (category, in100 band) -> words
    cat0_ntok = collections.Counter()     # (category, n_tokens) -> words
    cat0_examples = collections.defaultdict(collections.Counter)
    in100_0 = collections.Counter()       # never-seen: train_100M count band -> words
    in100_0_first = collections.Counter() # same, first occurrences only
    rare_cat = collections.Counter()      # 1-63 band: category of the word itself (name-like etc.)
    rare_c10 = collections.Counter()      # 1-63: exact c10 -> words
    rare_c100band = collections.Counter() # 1-63: c100 band -> words
    rare_ntok = collections.Counter()
    rare_examples = collections.defaultdict(collections.Counter)
    types0 = set()
    types_rare = set()
    ntok_band = collections.Counter()     # band -> total tokens (surface word tokenized alone with a leading space)
    nwords_band = collections.Counter()
    header_words = 0

    def c100_band(c: int) -> str:
        if c == 0:
            return "0"
        if c <= 9:
            return "1-9"
        if c <= 63:
            return "10-63"
        if c <= 630:
            return "64-630"
        return "631+"

    enc_cache: dict[str, int] = {}

    def ntoks(surface: str) -> int:
        k = enc_cache.get(surface)
        if k is None:
            k = len(tok.encode(" " + surface, add_special_tokens=False).ids)
            enc_cache[surface] = k
        return k

    for s in SOURCES:
        p = root / "dev" / f"{s}.dev"
        lines = p.read_text(encoding="utf-8").split("\n")
        for did, dl in docs(lines, s):
            seen_doc: set[str] = set()
            recent: collections.deque = collections.deque(maxlen=300)
            recent_set: collections.Counter = collections.Counter()
            for ln in dl:
                if is_header(ln):
                    header_words += len(ln.split())
                    continue
                for w in ln.split():
                    n = norm(w)
                    c = c10.get(n, 0)
                    b = band(c)
                    tot[b] += 1
                    tot_src[(s, b)] += 1
                    first_doc = n not in seen_doc
                    rep[(b, "first" if first_doc else "repeat_doc")] += 1
                    rep300[(b, "first" if recent_set[n] == 0 else "repeat")] += 1
                    seen_doc.add(n)
                    if len(recent) == recent.maxlen:
                        old = recent[0]
                        recent_set[old] -= 1
                        if recent_set[old] <= 0:
                            del recent_set[old]
                    recent.append(n)
                    recent_set[n] += 1
                    if b in ("0", "1-63"):
                        k = ntoks(w)
                        ntok_band[b] += k
                        nwords_band[b] += 1
                    if c == 0:
                        cat, det = classify(w, n, c10, c100, tok)
                        cat0[cat] += 1
                        cat0_src[(s, cat)] += 1
                        cb = c100_band(c100.get(n, 0))
                        cat0_in100[(cat, cb)] += 1
                        in100_0[cb] += 1
                        if first_doc:
                            in100_0_first[cb] += 1
                        cat0_ntok[(cat, min(ntoks(w), 8))] += 1
                        if len(cat0_examples[cat]) < 400:
                            cat0_examples[cat][f"{w} [{det}]" if det else w] += 1
                        types0.add(n)
                    elif c <= 63:
                        rare_c10[c] += 1
                        rare_c100band[c100_band(c100.get(n, 0))] += 1
                        rare_ntok[min(ntoks(w), 8)] += 1
                        types_rare.add(n)
                        cap = w.strip(PUNCT)[:1].isupper()
                        if any(ch.isdigit() for ch in n):
                            rc = "number/digits"
                        elif cap:
                            rc = "capitalized"
                        else:
                            rc = "lowercase"
                        rare_cat[rc] += 1
                        if len(rare_examples[rc]) < 300:
                            rare_examples[rc][w] += 1
        print(f"dev {s} done", flush=True)

    N = sum(tot.values())
    res = {
        "dev_words": N, "header_words_skipped": header_words,
        "band_share": {b: tot[b] / N for b in ("0", "1-63", "64-8k", "8k+")},
        "band_words": dict(tot),
        "band_share_by_source": {s: {b: tot_src[(s, b)] / max(1, sum(tot_src[(s, bb)] for bb in ("0", "1-63", "64-8k", "8k+")))
                                     for b in ("0", "1-63", "64-8k", "8k+")} for s in SOURCES},
        "source_words": {s: sum(tot_src[(s, bb)] for bb in ("0", "1-63", "64-8k", "8k+")) for s in SOURCES},
        "repeat_share_in_document": {b: rep[(b, "repeat_doc")] / max(1, rep[(b, "first")] + rep[(b, "repeat_doc")])
                                     for b in ("0", "1-63", "64-8k", "8k+")},
        "repeat_share_prev300": {b: rep300[(b, "repeat")] / max(1, rep300[(b, "first")] + rep300[(b, "repeat")])
                                 for b in ("0", "1-63", "64-8k", "8k+")},
        "tokens_per_word": {b: ntok_band[b] / max(1, nwords_band[b]) for b in ("0", "1-63")},
        "never_seen": {
            "types": len(types0), "occurrences": tot["0"],
            "category_share": {k: v / tot["0"] for k, v in cat0.most_common()},
            "category_by_source": {s: {k: cat0_src[(s, k)] for k in cat0} for s in SOURCES},
            "in_train_100M_band_share": {k: v / tot["0"] for k, v in sorted(in100_0.items())},
            "in_train_100M_band_share_first_occurrences": {k: v / max(1, sum(in100_0_first.values())) for k, v in sorted(in100_0_first.items())},
            "category_x_in100": {f"{k[0]} | c100 {k[1]}": v for k, v in sorted(cat0_in100.items())},
            "category_x_ntok": {f"{k[0]} | {k[1]} tok": v for k, v in sorted(cat0_ntok.items())},
            "examples": {k: [w for w, _ in v.most_common(40)] for k, v in cat0_examples.items()},
        },
        "rare_1_63": {
            "types": len(types_rare), "occurrences": tot["1-63"],
            "c10_distribution": {str(k): v for k, v in sorted(rare_c10.items())},
            "c100_band_share": {k: v / tot["1-63"] for k, v in sorted(rare_c100band.items())},
            "surface_kind_share": {k: v / tot["1-63"] for k, v in rare_cat.most_common()},
            "ntok_distribution": {str(k): v for k, v in sorted(rare_ntok.items())},
            "examples": {k: [w for w, _ in v.most_common(40)] for k, v in rare_examples.items()},
        },
    }
    (out / "rare_words_dev.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("never_seen", "rare_1_63")}, indent=1))
    ns = res["never_seen"]
    print("NEVER SEEN categories:", json.dumps(ns["category_share"], indent=1))
    print("NEVER SEEN in train_100M bands:", json.dumps(ns["in_train_100M_band_share"], indent=1))
    print("NEVER SEEN in train_100M bands (first occ):", json.dumps(ns["in_train_100M_band_share_first_occurrences"], indent=1))
    print("NEVER SEEN category x c100:", json.dumps(ns["category_x_in100"], indent=1))
    print("NEVER SEEN category x ntok:", json.dumps(ns["category_x_ntok"], indent=1))
    for k, v in ns["examples"].items():
        print(f"  {k}: {v[:25]}")
    r = res["rare_1_63"]
    print("RARE 1-63:", json.dumps({k: v for k, v in r.items() if k != "examples"}, indent=1))
    for k, v in r["examples"].items():
        print(f"  {k}: {v[:25]}")


if __name__ == "__main__":
    main()
