"""Official BabyLM 2025 strict-small GPT-2 baseline, single GPU, faithful port.

Source: github.com/momergul/babylm-gpt2-baseline @ ref-0492 (training.py, data_utils.py, models.py, config.yaml)
plus the hyperparameter table of hf.co/BabyLM-community/babylm-baseline-10m-gpt2 (21,300 steps, 213 warmup).
Model config and tokenizer are the published checkpoint's own files (the repo does not ship ./configs or ./tokenizers).

Faithful details kept from the official code:
- each domain file is read with readlines() and joined with ' ', tokenised whole, cut into 512-token chunks;
  every datapoint is [bos] + chunk + [eos]; batches padded with eos; targets masked where the INPUT is eos,
  except position 0 (data_utils.get_collate_fn);
- AdamW lr 5e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0 (config.yaml) on non-LayerNorm non-bias params;
  cosine schedule with warmup (213 / 21,300 steps); grad clip 1.0; bf16 autocast; batch 16; 10 epochs, shuffle.
- checkpoints at 1M..9M words during epoch 0 (tenths of the epoch, as the official strict_small branch) and
  at the end of every epoch (= 10M, 20M, ... 100M words).
Additions (do not change training): held-out dev loss on a fixed slice of the official dev split, CSV curve, HF export.
"""
import argparse, csv, json, math, os, random, time, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

DOMAINS = ['bnc_spoken', 'childes', 'gutenberg', 'open_subtitles', 'simple_wiki', 'switchboard']
HERE = os.path.dirname(os.path.abspath(__file__))


def tokenize_split(tok, folder, suffix, chunk):
    data, n_words = [], 0
    for d in DOMAINS:
        with open(os.path.join(folder, f'{d}.{suffix}')) as f:
            lines = f.readlines()
        n_words += sum(len(l.split()) for l in lines)
        ids = tok(text=[' '.join(lines)])['input_ids'][0]
        data += [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    return data, n_words


def collate(batch, bos, eos):
    toks = pad_sequence([torch.LongTensor([bos] + b + [eos]) for b in batch], padding_value=eos, batch_first=True)
    inp, tgt = toks[:, :-1], toks[:, 1:]
    mask = inp != eos
    mask[:, 0] = 1
    return inp, tgt, mask


def param_groups(model, wd):
    def names(m):
        out = []
        for n, c in m.named_children():
            out += [f'{n}.{x}' for x in names(c) if not isinstance(c, tuple(ALL_LAYERNORM_LAYERS))]
        return out + list(m._parameters.keys())
    decay = [n for n in names(model) if 'bias' not in n]
    return [{'params': [p for n, p in model.named_parameters() if n in decay], 'weight_decay': wd},
            {'params': [p for n, p in model.named_parameters() if n not in decay], 'weight_decay': 0.0}]


@torch.no_grad()
def dev_loss(model, dev, bos, eos, device, bs=32):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(dev), bs):
        inp, tgt, mask = (t.to(device) for t in collate(dev[i:i + bs], bos, eos))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(inp).logits
        nll = F.cross_entropy(logits.float().transpose(1, 2), tgt, reduction='none')
        tot += (nll * mask).sum().item(); n += mask.sum().item()
    model.train()
    return tot / n


def save_hf(model, tok, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True)
    ap.add_argument('--dev_dir', required=True)
    ap.add_argument('--base', default=os.path.join(HERE, 'base'), help='published config.json + tokenizer files')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--num_training_steps', type=int, default=0, help='0 = epochs x steps/epoch (published: 21,300)')
    ap.add_argument('--warmup_frac', type=float, default=0.01, help='published: 213 of 21,300 steps')
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--dev_chunks', type=int, default=2048, help='fixed dev slice: first N chunks in domain order')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--dev_every', type=int, default=500)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    bos, eos = tok.bos_token_id, tok.eos_token_id
    t0 = time.time()
    train, n_words = tokenize_split(tok, a.train_dir, 'train', a.chunk)
    dev_all, _ = tokenize_split(tok, a.dev_dir, 'dev', a.chunk)
    # fixed, deterministic dev slice spread over all domains: every k-th chunk
    k = max(1, len(dev_all) // a.dev_chunks)
    dev = dev_all[::k][:a.dev_chunks]
    n_tok = sum(len(c) for c in train)
    if a.smoke:
        train, dev = train[:8], dev[:2]
        a.epochs, a.batch_size, a.log_every, a.dev_every = 1, 2, 1, 2
    print(f'tokenised in {time.time()-t0:.0f}s: {len(train)} train chunks, {n_tok} tokens, {n_words} words; dev slice {len(dev)} chunks', flush=True)

    config = GPT2Config.from_pretrained(a.base)
    model = GPT2LMHeadModel(config).to(device)
    opt = torch.optim.AdamW(param_groups(model, a.weight_decay), lr=a.lr, eps=1e-8, betas=(0.9, 0.999))
    g = torch.Generator(); g.manual_seed(a.seed)
    loader = torch.utils.data.DataLoader(train, batch_size=a.batch_size, shuffle=True, generator=g,
                                         collate_fn=lambda b: collate(b, bos, eos))
    steps_per_epoch = len(loader)
    # The official code fixes 21,300 = 10 x 2,130 steps for its own (cleaned) copy of the data; the 2025 OSF files
    # give a different chunk count, so the cosine horizon is set to the real step count (a fixed 21,300 would make
    # the cosine rise again after its end).
    a.num_training_steps = a.num_training_steps or a.epochs * steps_per_epoch
    a.num_warmup_steps = max(1, round(a.warmup_frac * a.num_training_steps))
    sched = get_cosine_schedule_with_warmup(opt, a.num_warmup_steps, a.num_training_steps)
    words_per_step = n_words / steps_per_epoch
    meta = dict(vars(a), n_params=sum(p.numel() for p in model.parameters()), train_tokens_per_epoch=n_tok,
                train_words=n_words, steps_per_epoch=steps_per_epoch, device=str(device),
                torch=torch.__version__, gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None)
    json.dump(meta, open(os.path.join(a.out_dir, 'run_meta.json'), 'w'), indent=1)
    print(json.dumps(meta), flush=True)

    f = open(os.path.join(a.out_dir, 'curve.csv'), 'w', newline='')
    w = csv.writer(f); w.writerow(['step', 'tokens_seen', 'words_seen_est', 'epoch', 'lr', 'train_loss', 'dev_loss', 'elapsed_s'])
    step, tokens_seen, run_loss, run_tok, start = 0, 0, 0.0, 0, time.time()

    def log(dev=None):
        nonlocal run_loss, run_tok
        tr = run_loss / max(run_tok, 1) if run_tok else float('nan')
        w.writerow([step, tokens_seen, round(step * words_per_step), round(step / steps_per_epoch, 4),
                    sched.get_last_lr()[0], round(tr, 5), '' if dev is None else round(dev, 5), round(time.time() - start, 1)])
        f.flush(); run_loss, run_tok = 0.0, 0

    w0 = dev_loss(model, dev, bos, eos, device); log(w0)
    for epoch in range(a.epochs):
        tenth = steps_per_epoch // 10
        for i, (inp, tgt, mask) in enumerate(loader):
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(inp).logits
                lp = torch.gather(F.log_softmax(logits, dim=2), 2, tgt.unsqueeze(2)).squeeze(2)
            loss = -(lp * mask).sum() / mask.sum()
            loss.backward()
            if a.clip != -1:
                nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1; nt = mask.sum().item(); tokens_seen += nt
            run_loss += loss.item() * nt; run_tok += nt
            if epoch == 0 and i != 0 and tenth and i % tenth == 0 and i // tenth < 10:
                save_hf(model, tok, os.path.join(a.out_dir, 'ckpt', f'chck_{i // tenth}M'))
            if step % a.dev_every == 0:
                log(dev_loss(model, dev, bos, eos, device))
            elif step % a.log_every == 0:
                log()
        save_hf(model, tok, os.path.join(a.out_dir, 'ckpt', f'chck_{10 * (epoch + 1)}M'))
        print(f'epoch {epoch} done step {step} {time.time()-start:.0f}s', flush=True)
    final = dev_loss(model, dev, bos, eos, device); log(final)
    save_hf(model, tok, os.path.join(a.out_dir, 'final'))
    json.dump({'final_dev_loss': final, 'steps': step, 'tokens_seen': tokens_seen, 'train_s': time.time() - start},
              open(os.path.join(a.out_dir, 'result.json'), 'w'), indent=1)
    print('final dev loss', final, flush=True)


if __name__ == '__main__':
    main()
