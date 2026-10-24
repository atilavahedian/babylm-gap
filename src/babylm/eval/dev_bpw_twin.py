"""Blank-line twin of the registered DEV metric (experimenter 04:22Z, critic ref-0020): dev_bpw.py with only the blank-line
handling changed to match the training stream.

    python dev_bpw_twin.py MODEL_DIR SPLIT_DIR [--slice curve|full] [--out twin.json]   (dev_bpw.py's flags)

Registered (dev_bpw.py): each file is one document and its blank lines are text, so after every blank-line-separated
document the model is scored on '\\n\\n' (two newline tokens), a pair the training stream never contains.
Training stream (babylm/data/corpus.py): blank lines are dropped and every document starts with <|endoftext|> (BOS).
Twin: each run of blank (whitespace-only) lines, with the line break that ends the line before it, becomes one BOS
token, scored like every other token; blank lines at a file's start or end are dropped (its final line break stays);
each document is tokenized on its own, as prepare.py does. Everything else is dev_bpw.py itself, imported and run
unchanged (text, words, windows, fp32, report): only its tokenizer call is wrapped. Words are unchanged (blank lines
hold none); switchboard has no blank line, so its score is the registered one. The JSON adds 'protocol' and the
number of separators per file.

Second number, never-seen characters (critic ref-0015, the ex_markup pattern of the ruler's dev_bpw.py): 'ex_unseen_chars'
drops every whitespace word holding one of the 85 DEV characters that never occur in train_10M (UNSEEN_CHARS =
babylm/tools/unseen_chars_list.json @ ref-0016, set(dev) - set(train_10M) of the raw text), together with every token
whose text span touches such a word, from both the NLL and the word count (spans from the fast tokenizer's offsets;
whitespace-only tokens and the BOS separators are kept). To get per-token losses, dev_bpw.nll_of is replaced by
nll_tokens below: the same windows, batches and fp32 arithmetic, so its total is dev_bpw's to the bit (tested), and the
headline is unchanged. Slow tokenizer: the headline as before, ex_unseen_chars n/a.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

LEAD = re.compile(r'\A(?:[^\S\n]*\n)+')                  # blank lines at the start of a file
TRAIL = re.compile(r'\n(?:[^\S\n]*\n)*[^\S\n]*\Z')       # final line break plus any blank lines after it
SEP = re.compile(r'\n(?:[^\S\n]*\n)+')                   # a line break plus the run of blank lines after it
PROTOCOL = 'blank-line twin: each run of blank lines -> one scored BOS; documents tokenized on their own'
UNSEEN_CHARS = ('>\x92«»ÿĞĦĩĶťŷƒǂǎǐɕɘɝɹʔʕʹ˧ḌḻṉṑṟṢṮṯấẩậắẽếểịốồớởợụừự‏‐‡‭‰›⁎⁣₫℞↗⇗∆∇∙√∞∪≤⌈⌉⌊⌋▔○●★☺☻♙♟⚥⨳⸸'
                '꧈️🆒😎')
UNSEEN_REF = 'babylm/tools/unseen_chars_list.json @ ref-0016'


def documents(text):
    """The blank-line-separated documents of a registered text (the last keeps the file's final line break)."""
    return SEP.split(TRAIL.sub('\n', LEAD.sub('', text)))


def unseen_mask(text, offsets, chars=UNSEEN_CHARS):
    """(token mask, word count) for whitespace words holding one of chars; a token is masked if any character of its
    offset span lies in such a word (dev_bpw.markup_mask's rule)."""
    word_of = np.full(len(text), -1, dtype=np.int64)
    bad = []
    for k, m in enumerate(re.finditer(r'\S+', text)):
        word_of[m.start():m.end()] = k
        bad.append(any(c in chars for c in m.group()))
    bad = np.array(bad + [False], dtype=bool)   # index -1 (whitespace) -> not bad
    mask = np.array([bool(bad[word_of[a:b]].any()) if b > a else False for a, b in offsets], dtype=bool)
    return mask, int(bad[:-1].sum())


class TwinTokenizer:
    """The model's tokenizer, except that a call on a whole file returns its documents' ids joined by BOS (and, with a
    fast tokenizer, records which of those tokens fall in never-seen-character words)."""

    def __init__(self, tok):
        self.tok, self.separators, self.unseen = tok, [], []
        self.sep_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    def __getattr__(self, name):
        return getattr(self.tok, name)

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        docs, ids, masks, words = documents(text), [], [], 0
        fast = getattr(self.tok, 'is_fast', False)
        for k, d in enumerate(docs):
            enc = self.tok(d, add_special_tokens=False, **({'return_offsets_mapping': True} if fast else {}))
            ids += ([self.sep_id] if k else []) + enc['input_ids']
            if fast:
                m, w = unseen_mask(d, enc['offset_mapping'])
                masks += [np.zeros(1 if k else 0, dtype=bool), m]
                words += w
        self.separators.append(len(docs) - 1)
        self.unseen.append((np.concatenate(masks), words) if fast else None)
        return {'input_ids': ids}


def nll_tokens(model, ids, start_id, device, bs=16, CTX=512, STRIDE=256, vocab_mask=0):
    """dev_bpw.nll_of (with the A2 branch's vocab_mask) returning (total, per-token -log p): the windows, batches and
    the total's fp32 arithmetic are dev_bpw's own lines, so the total is identical."""
    import torch
    import torch.nn.functional as F
    n = len(ids)
    jobs = []  # (window tokens, number of scored tail tokens, window start in ids)
    k = 0
    while True:
        lo = k * STRIDE
        if lo >= n and k > 0: break
        win = ids[lo: lo + CTX - 1]
        score = len(win) if k == 0 else max(0, len(win) - (CTX - 1 - STRIDE))
        if score: jobs.append(([start_id] + win, score, lo))
        if lo + CTX - 1 >= n: break
        k += 1
    total, scored, tok_nll = 0.0, 0, np.full(n, np.nan)
    with torch.no_grad():
        for i in range(0, len(jobs), bs):
            batch = jobs[i:i + bs]
            L = max(len(w) for w, _, _ in batch)
            inp = torch.full((len(batch), L), start_id, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (w, _, _) in enumerate(batch):
                inp[j, :len(w)] = torch.tensor(w); att[j, :len(w)] = 1
            inp, att = inp.to(device), att.to(device)
            out = model(input_ids=inp, attention_mask=att)
            logits = (out[0] if isinstance(out, tuple) else out['logits']).float()
            if vocab_mask:
                logits[..., vocab_mask:] = float('-inf')
            lp = F.log_softmax(logits[:, :-1], -1).gather(-1, inp[:, 1:, None]).squeeze(-1)
            for j, (w, s, lo) in enumerate(batch):
                t = len(w) - 1  # predicted positions 1..len(w)-1; lp[j, q] is the loss of text token lo + q
                total -= lp[j, t - s:t].sum().item(); scored += s
                tok_nll[lo + t - s: lo + t] = -lp[j, t - s:t].double().cpu().numpy()
    assert scored == n and not np.isnan(tok_nll).any(), (scored, n)
    return total, tok_nll


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dev_bpw as R  # the registered protocol, run unchanged below except for nll_of's per-token twin

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--out')
    ap.add_argument('--line_join', default='newline')
    a, _ = ap.parse_known_args()
    if a.line_join != 'newline':
        raise SystemExit('the twin matches the harness training stream (--line_join newline) only')
    real, made, unseen = R.AutoTokenizer, [], []

    class AutoTwin:
        @staticmethod
        def from_pretrained(*args, **kw):
            made.append(TwinTokenizer(real.from_pretrained(*args, **kw)))
            return made[-1]

    def nll_of(model, ids, *args, **kw):  # called once per file, right after that file's tokenizer call
        total, tn = nll_tokens(model, ids, *args, **kw)
        u = made[0].unseen[len(unseen)]
        if u is None:
            unseen.append(None)
        else:
            mask, words = u
            assert len(mask) == len(ids), (len(mask), len(ids))
            unseen.append({'unseen_chars_words': words, 'unseen_chars_tokens': int(mask.sum()),
                           'unseen_chars_nats': float(tn[mask].sum())})
        return total

    R.AutoTokenizer, R.nll_of = AutoTwin, nll_of
    R.main()
    fast = all(u is not None for u in unseen)
    na = {'unseen_chars_words': None, 'unseen_chars_tokens': None, 'unseen_chars_nats': None,
          'ex_unseen_chars_nats_per_word': None, 'ex_unseen_chars_bits_per_word': None}
    ex = dict(na, unseen_chars_list=UNSEEN_REF, unseen_chars_count=len(UNSEEN_CHARS),
              unseen_chars_sha256=hashlib.sha256(UNSEEN_CHARS.encode()).hexdigest())
    if a.out:  # per_domain keeps dev_bpw's file order, the order of the tokenizer calls
        d = json.load(open(a.out))
        WU = NU = 0
        for f, n, u in zip(d['per_domain'], made[0].separators, unseen):
            p = d['per_domain'][f]
            p['separators'] = n
            if fast:
                nll = p['nats_per_word'] * p['words']
                npw = (nll - u['unseen_chars_nats']) / (p['words'] - u['unseen_chars_words'])
                p.update(u, ex_unseen_chars_nats_per_word=npw, ex_unseen_chars_bits_per_word=npw / math.log(2))
                WU += u['unseen_chars_words']; NU += u['unseen_chars_nats']
            else:
                p.update(na)
        if fast:
            N = d['nats_per_word'] * d['words']  # dev_bpw's own total, not the per-file products
            ex.update(unseen_chars_words=WU, unseen_chars_tokens=sum(u['unseen_chars_tokens'] for u in unseen),
                      unseen_chars_nats=NU, ex_unseen_chars_nats_per_word=(N - NU) / (d['words'] - WU),
                      ex_unseen_chars_bits_per_word=(N - NU) / (d['words'] - WU) / math.log(2))
        d['protocol'], d['separators'] = PROTOCOL, sum(made[0].separators)
        d.update(ex)
        json.dump(d, open(a.out, 'w'), indent=1)
        print(json.dumps({k: d[k] for k in ['nats_per_word'] + list(ex)}, indent=1))


if __name__ == '__main__':
    main()
