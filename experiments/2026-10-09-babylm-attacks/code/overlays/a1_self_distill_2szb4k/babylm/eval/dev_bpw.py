"""THE held-out metric protocol (critic rule ref-0053): DEV nats and bits per whitespace word, identical for every model.

    python dev_bpw.py MODEL_DIR SPLIT_DIR [--slice curve|full] [--out result.json]

Protocol (fixed; independent of each model's training format):
- Text: each of the 6 domain files of the split in sorted name order, read as ONE contiguous document per file,
  lines joined exactly as in the file (newlines kept). Slice 'curve' = the first 200,000 whitespace words of each file
  (whole lines, cut at the first line that reaches it); slice 'full' = whole files.
- Words: len(text.split()) of the scored text (whitespace words). Every token of the text is scored exactly once.
- Tokens: the model's own tokenizer, no special tokens added inside the text. Each window starts with the tokenizer's
  BOS id if it has one (else EOS), which is never scored.
- Windows: context 512 tokens including the BOS slot (the shortest max-position of the compared models), stride 256:
  window k covers text tokens [256k, 256k+511); its first window scores all its tokens, later windows score only their
  last 256 tokens, so each scored token sees >= 255 tokens of left context (except at a file start).
- Model: AutoModelForCausalLM(trust_remote_code=True), eval mode, fp32 weights, no autocast. Loss = -log p(token).
- Short-context rule (critic ref-0054): a model trained with context under 512 tokens is evaluated at its trained context
  (--ctx N, stride N/2) and flagged short_context=true; loss comparisons need equal effective context.
- Whitespace (critic ref-0055): headline = 'collapse', every run of whitespace inside a line becomes one space (newlines
  kept; GPT-BERT's training format never contains tabs or double spaces); 'raw' (the files as they are) is secondary.
  Word counts are identical.
- Line format (critic ref-0051): each model is scored in its own training line format, committed per recipe in
  babylm/recipes/<r>/format.json before scoring and never chosen by score: --line_join newline (GPT-BERT and every model
  we train: collapse + plain newlines) or newline_space (official GPT-2: ' '.join(f.readlines()), raw whitespace);
  --tab_space lists sources whose speaker-label tabs become spaces. Critic ref-0056: the training code decides the
  headline (GPT-2: none); the instruments' train-text probe @ ref-0057 (GPT-2: childes) is a second labelled number.
- Report: nats/word = total NLL / words; bits/word = nats/word / ln 2; plus per-domain values and nats/token.
Never run on the test split unless the critic calls it.
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CTX, STRIDE, CURVE_WORDS = 512, 256, 200_000


def load_text(path, slice_, ws='collapse', line_join='newline', tab_space=()):
    """line_join 'newline': lines as in the file; 'newline_space': ' '.join(f.readlines()) (official GPT-2 baseline).
    tab_space: sources (file-name prefixes) whose tabs become single spaces (only matters with ws='raw')."""
    lines = open(path, encoding='utf-8').readlines()          # keeps each line's '\n'
    if any(os.path.basename(path).startswith(s) for s in tab_space):
        lines = [l.replace('\t', ' ') for l in lines]
    if ws == 'collapse':  # whitespace inside each line -> single spaces (word count unchanged)
        lines = [' '.join(l.split()) + ('\n' if l.endswith('\n') else '') for l in lines]
    if slice_ == 'curve':
        out, n = [], 0
        for l in lines:
            out.append(l); n += len(l.split())
            if n >= CURVE_WORDS:
                out[-1] = out[-1].rstrip('\n')   # a cut ends at that line's text (as before)
                break
        lines = out
    return ('' if line_join == 'newline' else ' ').join(lines)


@torch.no_grad()
def nll_of(model, ids, start_id, device, bs=16, CTX=CTX, STRIDE=STRIDE):
    """Sum of -log p over all tokens in ids with the sliding-window protocol."""
    n = len(ids)
    jobs = []  # (window tokens, number of scored tail tokens)
    k = 0
    while True:
        lo = k * STRIDE
        if lo >= n and k > 0: break
        win = ids[lo: lo + CTX - 1]
        score = len(win) if k == 0 else max(0, len(win) - (CTX - 1 - STRIDE))
        if score: jobs.append(([start_id] + win, score))
        if lo + CTX - 1 >= n: break
        k += 1
    total, scored = 0.0, 0
    for i in range(0, len(jobs), bs):
        batch = jobs[i:i + bs]
        L = max(len(w) for w, _ in batch)
        inp = torch.full((len(batch), L), start_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (w, _) in enumerate(batch):
            inp[j, :len(w)] = torch.tensor(w); att[j, :len(w)] = 1
        inp, att = inp.to(device), att.to(device)
        out = model(input_ids=inp, attention_mask=att)
        logits = (out[0] if isinstance(out, tuple) else out['logits']).float()
        lp = F.log_softmax(logits[:, :-1], -1).gather(-1, inp[:, 1:, None]).squeeze(-1)
        for j, (w, s) in enumerate(batch):
            t = len(w) - 1  # predicted positions 1..len(w)-1
            total -= lp[j, t - s:t].sum().item(); scored += s
    assert scored == n, (scored, n)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir'); ap.add_argument('split_dir')
    ap.add_argument('--slice', choices=['curve', 'full'], default='full')
    ap.add_argument('--out')
    ap.add_argument('--ws', choices=['raw', 'collapse'], default='collapse',
                    help="'collapse' (headline, critic ref-0055): whitespace runs inside a line -> one space; 'raw' = secondary")
    ap.add_argument('--line_join', choices=['newline', 'newline_space'], default='newline',
                    help="the model's own training line format (critic ref-0051), read from babylm/recipes/<r>/format.json")
    ap.add_argument('--tab_space', default='', help="comma list of sources whose tabs become spaces (critic ref-0056), e.g. childes")
    ap.add_argument('--ctx', type=int, default=CTX, help='only below 512 for short-context models (stride = ctx/2)')
    a = ap.parse_args()
    assert 'test' not in os.path.basename(os.path.normpath(a.split_dir)), 'test split is off-limits unless the critic calls it'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    # no torch_dtype kwarg: GPT-BERT's config class cannot serialise a dtype object; cast after loading instead
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True).float().to(device).eval()
    start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    t0, res, T, W, N = time.time(), {}, 0, 0, 0.0
    for f in sorted(os.listdir(a.split_dir)):
        text = load_text(os.path.join(a.split_dir, f), a.slice, a.ws, a.line_join, tuple(x for x in a.tab_space.split(',') if x))
        ids = tok(text, add_special_tokens=False)['input_ids']
        nll = nll_of(model, ids, start_id, device, CTX=a.ctx, STRIDE=a.ctx // 2)
        w = len(text.split())
        res[f] = {'words': w, 'tokens': len(ids), 'nats_per_word': nll / w, 'bits_per_word': nll / w / math.log(2)}
        T += len(ids); W += w; N += nll
    summary = {'model_dir': a.model_dir, 'split_dir': a.split_dir, 'slice': a.slice, 'ctx': a.ctx, 'stride': a.ctx // 2,
               'short_context': a.ctx < CTX, 'ws': a.ws, 'line_join': a.line_join, 'tab_space': a.tab_space,
               'words': W, 'tokens': T, 'nats_per_word': N / W, 'bits_per_word': N / W / math.log(2),
               'nats_per_token': N / T, 'per_domain': res, 'seconds': round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=1))
    if a.out: json.dump(summary, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
