"""Single-GPU port of the official BabyLM 2025 strict-small GPT-BERT baseline (causal-focus by default).

Sources: github.com/ltgoslo/gpt-bert @ ref-0493 pretraining/train_10m.py (+ model_extra.py, dataset.py, lamb.py, utils.py,
copied verbatim into ./official) and the hyperparameter table of
hf.co/BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus (README.md). The repo HEAD carries 2024 defaults
(vocab 8,192, LR 1.41e-2, global batch 32,768 sequences with a 4x batch ramp, 7,812 steps, seq switch at 70/90%);
every value here is the 2025 card value, set explicitly and recorded in run_meta.json.

How the multi-GPU hybrid is emulated on one GPU
  Official: W ranks; a fixed fraction of ranks trains MNTP (masked), the rest CLM (causal); every rank's loss is the mean
  over its own targets; DDP averages gradients over ranks => grad of  m*L_masked + (1-m)*L_causal  (m = masked share).
  Here: each step builds ONE batch of B = batch_tokens/seq_len sequences, the first B*m from MaskedDataset and the rest
  from CausalDataset (official classes, unchanged), and calls the official hybrid forward
  Bert.forward(..., num_masked=B*m, ratio=m), which returns m*L_masked + (1-m)*L_causal (and the same mix of z-losses).

FLAG SEMANTICS: --hybrid_numerator/--hybrid_denominator is the CAUSAL share here (15/16 = causal-focus, the published
model). In the official train_10m.py the numerator counts MASKED ranks. Both are printed at start-up.

Word exposure: a token of the tokenised corpus starts a whitespace word iff its string starts with the byte-level space
(the count equals str.split() words of train_10M and dev exactly). Words consumed per batch = word-start tokens among the
data tokens the model actually saw (causal: all targets; masked: visible inputs + masked targets). BabyLM checkpoints
(chck_1M..chck_9M, chck_10M..chck_90M, chck_100M..chck_1000M) are triggered by this counter.

CSV columns: dev_loss_causal / dev_loss_masked = fixed 1M-token dev subset (every --eval_every steps);
dev_bpw_curve = critic protocol eval/dev_bpw.py (--slice curve, ctx --curve_ctx) via its own nll_of/load_text;
dev_dump_loss_causal, dev_dump_bpw, train_sample_loss_causal, train_sample_bpw = per-token dumps (epoch checkpoints, end).
"""
import argparse, copy, csv, json, math, os, platform, queue, random, socket, subprocess, sys, threading, time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "official"))
from model_extra import Bert                                   # noqa: E402  (official, verbatim)
from lamb import Lamb                                          # noqa: E402
from utils import cosine_schedule_with_warmup_cooldown         # noqa: E402
from dataset import MaskedDataset, CausalDataset, SpanMaskingStrategy  # noqa: E402
from tokenizers import Tokenizer                               # noqa: E402
import export_hf                                               # noqa: E402
import scoring                                                 # noqa: E402
from prepare_data import word_start_table, newline_table, sha256_file  # noqa: E402

DEV_BPW_PY = os.path.normpath(os.path.join(HERE, "..", "..", "eval", "dev_bpw.py"))
OFFICIAL_COMMIT = "ref-0493"
REPO_2024_DEFAULTS = {"vocab_size": 8192, "learning_rate": 1.41e-2, "global_batch_size_sequences": 32768, "batch_reduction": 4,
                      "max_steps": 7812, "seq_switch_fracs": "0.7,0.9", "layer_norm_eps": 1e-5, "hybrid_masked_numerator": 15}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.path.join(HERE, "data"), help="output folder of prepare_data.py")
    p.add_argument("--train_dir", default="train_10M", help="train_10M or train_100M (a path's basename is used)")
    p.add_argument("--dev_text_dir", default="~/babylm/osf/text_data/dev/dev", help="raw dev files for dev_bpw.py curve")
    p.add_argument("--tokenizer_path", default=os.path.join(HERE, "hf_files", "tokenizer.json"))
    p.add_argument("--model_config", default=None, help="json with model fields; default = published small model")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    # hybrid objective (CAUSAL share; see module docstring)
    p.add_argument("--hybrid_numerator", type=int, default=15)
    p.add_argument("--hybrid_denominator", type=int, default=16)
    # 2025 model-card hyperparameters (explicit)
    p.add_argument("--epochs", type=float, default=10.0, help="max_steps = round(epochs * corpus_tokens / batch_tokens) (10M: 9,914)")
    p.add_argument("--max_steps", type=int, default=None, help="override the epoch-derived step count")
    p.add_argument("--seq_length", type=int, default=128, help="base sequence length (x2 and x4 later)")
    p.add_argument("--seq_switch_fracs", default="0.6,0.8", help="fractions of training where seq len doubles (card: 60/20/20)")
    p.add_argument("--batch_tokens", type=int, default=16384, help="tokens per optimizer step (constant across seq lengths)")
    p.add_argument("--learning_rate", type=float, default=0.007)
    p.add_argument("--warmup_proportion", type=float, default=0.016)
    p.add_argument("--cooldown_proportion", type=float, default=0.016)
    p.add_argument("--min_lr_factor", type=float, default=0.1, help="hard-coded 0.1 in train_10m.py")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--mask_p_start", type=float, default=0.3)
    p.add_argument("--mask_p_end", type=float, default=0.15)
    p.add_argument("--mask_random_p", type=float, default=0.1)
    p.add_argument("--mask_keep_p", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--optimizer_eps", type=float, default=1e-8)
    p.add_argument("--optimizer_beta1", type=float, default=0.9)
    p.add_argument("--optimizer_beta2", type=float, default=0.98)
    p.add_argument("--max_gradient", type=float, default=2.0)
    p.add_argument("--z_loss_weight", type=float, default=1e-4)
    p.add_argument("--n_special_tokens", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1, help="split each step into k micro-batches (memory only)")
    p.add_argument("--mixed_precision", default=True, action=argparse.BooleanOptionalAction, help="bf16 autocast (CUDA only)")
    # logging / eval
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=200, help="dev-subset eval every N steps and at checkpoints")
    p.add_argument("--dev_tokens", type=int, default=1_000_000, help="target tokens in the fixed dev subset")
    p.add_argument("--dev_seq_len", type=int, default=128)
    p.add_argument("--dev_batch", type=int, default=64)
    p.add_argument("--dev_seed", type=int, default=2025)
    p.add_argument("--bpw_every", type=int, default=1000, help="dev_bpw.py curve-slice eval every N steps (+ checkpoints, end); 0 = off")
    p.add_argument("--curve_ctx", type=int, default=128, help="ctx for the in-training dev_bpw curve (stride ctx/2); 128 = the context "
                   "every checkpoint was trained at (critic short-context rule); the final ctx-512 number comes from vm_job.sh")
    p.add_argument("--curve_ws", choices=["raw", "collapse"], default="collapse",
                   help="dev_bpw.py --ws for the curve; collapse = training format (raw tabs cost 13-23 nats each)")
    p.add_argument("--curve_words", type=int, default=None, help="testing: words per dev file for the curve (default dev_bpw's 200k)")
    # checkpoints / dumps
    p.add_argument("--ckpt_word_unit", type=float, default=1e6, help="checkpoints at 1..9, 10..90, 100..1000 units of words")
    p.add_argument("--dump_every_words", type=float, default=None, help="per-token dumps at checkpoints that are multiples "
                   "of this (default: one epoch = 10M words for train_10M, 100M for train_100M) and at the end")
    p.add_argument("--dumps", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--dump_ctx", type=int, default=128, help="scoring context incl. <s> (train sample chunks are 127 tokens)")
    p.add_argument("--dump_batch", type=int, default=128)
    p.add_argument("--dump_precision", choices=["fp32", "tf32", "bf16"], default="fp32")
    p.add_argument("--dump_limit_tokens", type=int, default=0, help="0 = full dev split / full train sample")
    p.add_argument("--resume", default=None, help="states/<chck>.pt written by this script")
    p.add_argument("--resume_states", choices=["all", "last"], default="all", help="keep every resumable state or only the newest")
    p.add_argument("--prefetch", default=None, action=argparse.BooleanOptionalAction, help="background batch thread (default: on for CUDA)")
    p.add_argument("--init_from_hf", default=None, help="load weights from an HF GPT-BERT dir (e.g. the published one)")
    p.add_argument("--eval_only", action="store_true", help="dev-subset eval + dumps of --init_from_hf (or the init) and exit")
    p.add_argument("--smoke", action="store_true", help="tiny CPU run: data slice, seq 16/32/64, 1024 tok/step, 24 steps")
    p.add_argument("--smoke_train_tokens", type=int, default=200_000)
    p.add_argument("--stop_after_step", type=int, default=None, help="testing: exit after this step (to test --resume)")
    args = p.parse_args(argv)
    args.train_dir = os.path.basename(os.path.normpath(args.train_dir))
    assert args.train_dir in ("train_10M", "train_100M"), args.train_dir
    if args.smoke:
        args.seq_length, args.batch_tokens, args.max_steps = 16, 1024, 24   # B = 64/32/16 seqs: divisible by 16
        args.log_every, args.eval_every, args.dev_tokens, args.dev_seq_len, args.dev_batch = 2, 8, 2048, 32, 32
        args.ckpt_word_unit, args.dump_every_words, args.dump_limit_tokens = 4000, 8000, 3000
        args.bpw_every, args.curve_ctx, args.curve_words = 12, 32, 300
    if args.dump_every_words is None:
        args.dump_every_words = 10e6 if args.train_dir == "train_10M" else 100e6
    return args


# ------------------------------------------------------------------------------------------------------------ helpers
def seq_len_for_step(step, args):
    """Card: base for the first 60% of steps, x2 for the next 20%, x4 for the last 20% (switch by step, not epoch)."""
    f1, f2 = (float(x) for x in args.seq_switch_fracs.split(","))
    frac = (step + 1) / args.max_steps  # same (global_step + 1) / max_steps test as train_10m.py:load_datasets
    if frac >= f2:
        return args.seq_length * 4
    if frac >= f1:
        return args.seq_length * 2
    return args.seq_length


def checkpoint_thresholds(unit):
    return [unit * k for k in list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 1001, 100))]


def ckpt_name(words, unit):
    k = int(round(words / unit))
    return f"chck_{k}M" if unit == 1e6 else f"chck_{k}x{int(unit)}w"


def git_info():
    try:
        sha = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--", HERE], capture_output=True, text=True, timeout=10).stdout.strip()
        return {"sha": sha or None, "recipe_dir_dirty_or_untracked": bool(dirty)}
    except Exception as e:
        return {"sha": None, "error": str(e)}


class IndexStream:
    """Shuffled index stream over a dataset; drop_last semantics like the official DataLoader(shuffle, drop_last)."""
    def __init__(self, n, seed):
        self.n, self.gen, self.pos, self.epoch, self.perm = n, torch.Generator().manual_seed(seed), n, -1, None

    def take(self, k):
        if self.pos + k > self.n:
            self.perm, self.pos, self.epoch = torch.randperm(self.n, generator=self.gen), 0, self.epoch + 1
        out = self.perm[self.pos:self.pos + k]
        self.pos += k
        return out.tolist()

    def state_dict(self):
        return {"gen": self.gen.get_state(), "perm": None if self.perm is None else self.perm.clone(), "pos": self.pos, "epoch": self.epoch}

    def load_state_dict(self, s):
        self.gen.set_state(s["gen"]); self.perm, self.pos, self.epoch = s["perm"], s["pos"], s["epoch"]


class HybridBatcher:
    """Produces the hybrid batch for each step from the official datasets. Owns ALL data randomness: the official dataset
    code draws from the global torch CPU RNG, so batch() swaps in a private RNG state (self.rng_state); data order is then
    independent of dropout RNG and exactly resumable. Also counts words / data tokens consumed and decides checkpoints,
    attaching a data-state snapshot to the batch at which a checkpoint fires (so the prefetch thread can run ahead)."""

    def __init__(self, args, tokenizer, train_path, ws_table):
        self.args, self.tok, self.path = args, tokenizer, train_path
        self.c_num, self.c_den = args.hybrid_numerator, args.hybrid_denominator
        self.ds, self.streams = {}, {}
        self.cls, self.pad = tokenizer.token_to_id("<s>"), tokenizer.token_to_id("<pad>")
        self.ws = torch.from_numpy(ws_table)
        self.rng_state = torch.Generator().manual_seed(args.seed).get_state()
        self.lock = threading.Lock()
        self.words = self.data_tokens = self.tokens = 0
        self.thresholds = checkpoint_thresholds(args.ckpt_word_unit)
        self.next_ckpt = 0

    def _get(self, kind, seq_len):
        key = (kind, seq_len)
        if key not in self.ds:
            cls = MaskedDataset if kind == "masked" else CausalDataset
            self.ds[key] = cls(self.path, self.tok, self.args, seq_len, None, None)  # rank=None: one process sees all data
            self.streams[key] = IndexStream(len(self.ds[key]), self.args.seed * 1000 + (1 if kind == "masked" else 2) * 100 + seq_len)
        return self.ds[key], self.streams[key]

    def batch(self, step):
        with self.lock:
            prev = torch.get_rng_state()
            torch.set_rng_state(self.rng_state)
            try:
                b = self._batch(step)
            finally:
                self.rng_state = torch.get_rng_state()
                torch.set_rng_state(prev)
        return b

    def _batch(self, step):
        a = self.args
        T = seq_len_for_step(step, a)
        B = a.batch_tokens // T
        assert (B * self.c_num) % self.c_den == 0, f"batch {B} not divisible into {self.c_num}/{self.c_den}"
        n_causal = B * self.c_num // self.c_den
        n_masked = B - n_causal
        items = []
        if n_masked:
            ds, st = self._get("masked", T)
            ds.set_global_step(step)
            items += [ds[i] for i in st.take(n_masked)]
        if n_causal:
            ds, st = self._get("causal", T)
            items += [ds[i] for i in st.take(n_causal)]
        input_ids = torch.stack([it[0] for it in items])            # [B, T]
        target_ids = torch.stack([it[1] for it in items])           # [B, T]
        attention_mask = torch.stack([it[2] for it in items])       # [B, T, T], True = blocked
        mask_p = torch.stack([it[3].float() for it in items[:n_masked]]).mean().item() if n_masked else 0.0
        # consumed data: causal = every non-ignored target; masked = original tokens of the visible input (masked
        # positions restored from their targets) + the masked target beyond the last input position.
        tg_c = target_ids[n_masked:]
        consumed = [tg_c[tg_c != -100]]
        if n_masked:
            inp, tg = input_ids[:n_masked].clone(), target_ids[:n_masked]
            m = tg[:, :-1] != -100
            inp[:, 1:][m] = tg[:, :-1][m]
            consumed += [inp[(inp != self.cls) & (inp != self.pad)], tg[:, -1][tg[:, -1] != -100]]
        consumed = torch.cat(consumed)
        n_data, n_words = int(consumed.numel()), int(self.ws[consumed].sum())
        self.words += n_words; self.data_tokens += n_data; self.tokens += B * T
        names = []
        while self.next_ckpt < len(self.thresholds) and self.words >= self.thresholds[self.next_ckpt]:
            names.append((ckpt_name(self.thresholds[self.next_ckpt], a.ckpt_word_unit), self.thresholds[self.next_ckpt]))
            self.next_ckpt += 1
        is_final = step == a.max_steps - 1
        return dict(input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask, num_masked=n_masked,
                    seq_len=T, mask_p=mask_p, n_data=n_data, n_words=n_words, words_total=self.words,
                    data_tokens_total=self.data_tokens, tokens_total=self.tokens, ckpts=names,
                    data_state=self._state_dict(T, step) if (names or is_final) else None)

    def _state_dict(self, T, step):
        """Called inside the RNG swap, so torch.get_rng_state() here IS the data RNG state after this batch."""
        ds_state, streams = {}, {}
        for key, ds in self.ds.items():
            if key[1] != T:
                continue  # sequence length only grows: datasets of earlier lengths are never used again
            st = {"counts": torch.cat(ds.counts).clone(), "ri_indices": ds.random_index.indices.clone(), "ri_index": int(ds.random_index.index)}
            if hasattr(ds, "mask_counts"):
                st["mask_counts"] = torch.cat(ds.mask_counts).clone()
            ds_state[key] = st
            streams[key] = self.streams[key].state_dict()
        return {"after_step": step + 1, "rng_state": torch.get_rng_state().clone(), "ds": ds_state, "streams": streams,
                "words": self.words, "data_tokens": self.data_tokens, "tokens": self.tokens, "next_ckpt": self.next_ckpt}

    def load_state_dict(self, s):
        with self.lock:
            prev = torch.get_rng_state()
            try:
                for key, st in s["ds"].items():
                    ds, stream = self._get(*key)
                    for t, v in zip(ds.counts, torch.split(st["counts"], [len(c) for c in ds.counts])):
                        t.copy_(v)
                    if "mask_counts" in st:
                        for t, v in zip(ds.mask_counts, torch.split(st["mask_counts"], [len(c) for c in ds.mask_counts])):
                            t.copy_(v)
                    ds.random_index.indices, ds.random_index.index = st["ri_indices"], st["ri_index"]
                    stream.load_state_dict(s["streams"][key])
            finally:
                torch.set_rng_state(prev)
            self.rng_state = s["rng_state"]
            self.words, self.data_tokens, self.tokens, self.next_ckpt = s["words"], s["data_tokens"], s["tokens"], s["next_ckpt"]


def prefetch_iter(batcher, start, stop, pin, depth=6):
    q = queue.Queue(maxsize=depth)

    def work():
        try:
            for s in range(start, stop):
                b = batcher.batch(s)
                if pin:
                    for k in ("input_ids", "target_ids", "attention_mask"):
                        b[k] = b[k].pin_memory()
                q.put(b)
        except BaseException as e:  # surface errors in the main thread
            q.put(e)
    threading.Thread(target=work, daemon=True).start()
    for _ in range(start, stop):
        b = q.get()
        if isinstance(b, BaseException):
            raise b
        yield b


# ------------------------------------------------------------------------------------------------------- dev subset
def build_dev_subset(args, tokenizer, dev_docs):
    """Fixed deterministic dev subset (identical for every run with the same --dev_* flags):
    all dev documents -> non-overlapping chunks of (dev_seq_len-1) tokens (tails >= 2 tokens kept) -> shuffled with
    random.Random(dev_seed) -> first chunks until >= dev_tokens tokens. Each chunk x gives
      causal: input [<s>]+x[:-1], targets x (all tokens, the first predicted from <s>), causal attention;
      masked (MNTP, = official ValidationDataset): span masking at p=0.15 (official SpanMaskingStrategy, 10% random,
              10% keep) with torch seed dev_seed, input = masked([<s>]+x)[:-1], target = masked positions shifted by one,
              bidirectional attention. Padding is attended by nobody."""
    L = args.dev_seq_len - 1
    chunks = [d[o:o + L] for d in dev_docs for o in range(0, len(d), L) if len(d) - o >= 2]
    random.Random(args.dev_seed).shuffle(chunks)
    sel, n = [], 0
    for c in chunks:
        if n >= args.dev_tokens:
            break
        sel.append(c.long()); n += len(c)
    cls, pad, mask_id = tokenizer.token_to_id("<s>"), tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<mask>")
    T = args.dev_seq_len
    masker = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, tokenizer.get_vocab_size(), mask_id)
    C_in, C_tg, C_m, M_in, M_tg, M_m = [], [], [], [], [], []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.dev_seed)
        for c in sel:
            k = len(c)
            seg = torch.cat([torch.tensor([cls]), c])                  # k+1
            att = torch.ones(T, T, dtype=torch.bool)
            att[:k, :k] = False                                        # True = blocked
            inp = torch.full((T,), pad); inp[:k] = seg[:-1]
            tg = torch.full((T,), -100); tg[:k] = seg[1:]
            C_in.append(inp); C_tg.append(tg); C_m.append(att | torch.ones(T, T, dtype=torch.bool).triu(1))
            ratios, repl = masker(seg)                                 # official ValidationDataset.apply_mask (0.15, strict '<')
            thr = torch.topk(ratios, max(1, int(ratios.size(0) * 0.15 + 0.5)), largest=False).values.max().item()
            m = ratios < thr
            tgt_full = torch.where(m, seg, torch.full_like(seg, -100))
            inp_full = torch.where(m, repl, seg)
            inp = torch.full((T,), pad); inp[:k] = inp_full[:-1]
            tg = torch.full((T,), -100); tg[:k] = tgt_full[1:]
            M_in.append(inp); M_tg.append(tg); M_m.append(att)
    dev = {"causal": (torch.stack(C_in), torch.stack(C_tg), torch.stack(C_m)),
           "masked": (torch.stack(M_in), torch.stack(M_tg), torch.stack(M_m))}
    info = {"chunks": len(sel), "tokens": n, "causal_targets": int((dev["causal"][1] != -100).sum()),
            "masked_targets": int((dev["masked"][1] != -100).sum())}
    return dev, info


@torch.no_grad()
def evaluate_subset(model, dev, device, batch):
    """Mean nats/token over the fixed dev subset, fp32 (no autocast), dropout off."""
    was_training = model.training
    model.eval()
    out = {}
    for kind, (I, Tg, M) in dev.items():
        tot, cnt = 0.0, 0
        for i in range(0, I.size(0), batch):
            ids, tg, m = I[i:i + batch].to(device), Tg[i:i + batch].to(device), M[i:i + batch].to(device)
            loss, _, _, n = model(ids.t(), m, tg.t())
            tot += loss.item() * n; cnt += n
        out[kind] = tot / cnt
    model.train(was_training)
    return out


# ------------------------------------------------------------------------------------- dev_bpw.py (critic protocol)
class HFCausalView(nn.Module):
    """Train-side Bert seen through the published GPTBERTForCausalLM interface (key-padding mask | causal triu), so
    eval/dev_bpw.py's nll_of can score the live model. export_hf.py --verify shows the logits are identical (0.0 diff)."""
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        mask = torch.ones(T, T, dtype=torch.bool, device=input_ids.device).triu(1).unsqueeze(0).expand(B, T, T)
        if attention_mask is not None:
            mask = mask | (attention_mask == 0)[:, None, :]
        h = self.bert.get_contextualized(input_ids.t(), mask)
        return (self.bert.classifier.nonlinearity(h).transpose(0, 1),)


class CurveBPW:
    """dev_bpw.py --slice curve (first 200k whitespace words per dev file, critic protocol) at ctx=curve_ctx, stride ctx/2."""
    def __init__(self, args):
        sys.path.insert(0, os.path.dirname(DEV_BPW_PY))
        import dev_bpw
        from transformers import AutoTokenizer
        self.mod, self.args = dev_bpw, args
        if args.curve_words:
            dev_bpw.CURVE_WORDS = args.curve_words
        tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "hf_files"))
        self.start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        self.files = []
        for f in sorted(os.listdir(args.dev_text_dir)):
            text = dev_bpw.load_text(os.path.join(args.dev_text_dir, f), "curve", args.curve_ws)
            self.files.append((f, tok(text, add_special_tokens=False)["input_ids"], len(text.split())))
        self.sha = sha256_file(DEV_BPW_PY)

    @torch.no_grad()
    def __call__(self, model, device):
        was = model.training
        model.eval()
        view, N, W, per = HFCausalView(model), 0.0, 0, {}
        try:
            for f, ids, w in self.files:
                nll = self.mod.nll_of(view, ids, self.start_id, device, bs=64, CTX=self.args.curve_ctx, STRIDE=self.args.curve_ctx // 2)
                per[f] = nll / w / math.log(2); N += nll; W += w
        finally:
            model.train(was)
        return {"bits_per_word": N / W / math.log(2), "nats_per_word": N / W, "words": W, "per_domain_bpw": per}


# ------------------------------------------------------------------------------------------------------------ dumps
class Dumper:
    """Per-token causal NLL dumps (scoring.py protocol: training-format text, chunks of dump_ctx-1 tokens after <s>) on
    the FULL dev split and on the fixed 1M-word train_10M sample. Static arrays (token ids int32, doc ids int32 = document
    index in data/dev_docs.npy, line ids int32 = 0-based line number inside its file, file ids int16, position-in-chunk
    int16) are written once to <out>/dumps/; per checkpoint <out>/dumps/<name>/{dev,train_sample}_nll.npy (float16) plus
    summary.json (mean nats/token, bits per whitespace word, per-file breakdown)."""

    def __init__(self, args, tokenizer, data_dir, out_dir, dev_docs):
        self.args, self.dir = args, os.path.join(out_dir, "dumps")
        os.makedirs(self.dir, exist_ok=True)
        ws, nl = word_start_table(tokenizer), newline_table(tokenizer)
        self.bos = tokenizer.token_to_id("<s>")
        self.sets = {}
        self.dev_files = json.load(open(os.path.join(data_dir, "dev_manifest.json")))["file_order"]
        dev_chunks = scoring.make_chunks(dev_docs, args.dump_ctx, args.dump_limit_tokens)
        self._add("dev", dev_docs, dev_chunks, np.load(os.path.join(data_dir, "dev_docs.npy")), ws, nl)
        tr_docs = torch.load(os.path.join(data_dir, "train_10M_tokenized.bin"))
        tr_chunks = [tuple(int(v) for v in c) for c in np.load(os.path.join(data_dir, "train_10M_sample1M_chunks.npy"))]
        assert all(c[2] <= args.dump_ctx - 1 for c in tr_chunks), "train sample chunks longer than dump_ctx-1"
        if args.dump_limit_tokens:
            keep, n = [], 0
            for c in tr_chunks:
                if n >= args.dump_limit_tokens:
                    break
                keep.append(c); n += c[2]
            tr_chunks = keep
        self._add("train_sample", tr_docs, tr_chunks, np.load(os.path.join(data_dir, "train_10M_docs.npy")), ws, nl)
        np.save(os.path.join(self.dir, "train_sample_lines.npy"), np.load(os.path.join(data_dir, "train_10M_sample1M_lines.npy")))
        json.dump({k: {"tokens": len(v["static"]["token_ids"]), "words": v["words"], "chunks": len(v["chunks"])} for k, v in self.sets.items()}
                  | {"dev_file_order": self.dev_files, "ctx": args.dump_ctx}, open(os.path.join(self.dir, "static_info.json"), "w"), indent=2)

    def _add(self, name, docs, chunks, table, ws, nl):
        st = scoring.chunk_token_arrays(docs, chunks, table, nl)
        for k, v in st.items():
            np.save(os.path.join(self.dir, f"{name}_{k}.npy"), v)
        self.sets[name] = {"docs": docs, "chunks": chunks, "static": st, "words": int(ws[st["token_ids"]].sum()), "wsm": ws[st["token_ids"]]}

    @torch.no_grad()
    def dump(self, model, name, device):
        was_training = model.training
        model.eval()
        prec = self.args.dump_precision
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
        fn = scoring.train_side_logits_fn(model)
        out_dir = os.path.join(self.dir, name)
        os.makedirs(out_dir, exist_ok=True)
        summ = {"precision": prec, "ctx": self.args.dump_ctx}
        try:
            for set_name, s in self.sets.items():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(prec == "bf16" and device.type == "cuda")):
                    nll = scoring.score_chunks(fn, s["docs"], s["chunks"], self.bos, device, self.args.dump_batch)
                np.save(os.path.join(out_dir, f"{set_name}_nll.npy"), nll.astype(np.float16))
                fids, wsm = s["static"]["file_ids"], s["wsm"]
                per_file = {}
                for f in np.unique(fids):
                    sel = fids == f
                    per_file[int(f)] = {"tokens": int(sel.sum()), "words": int(wsm[sel].sum()), "mean_nats": float(nll[sel].mean()),
                                        "bpw": scoring.bits_per_word(nll[sel], int(wsm[sel].sum()))}
                summ[set_name] = {"tokens": len(nll), "words": s["words"], "mean_nats": float(nll.mean(dtype=np.float64)),
                                  "bpw": scoring.bits_per_word(nll, s["words"]), "per_file": per_file}
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            model.train(was_training)
        json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
        return summ


# -------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    use_amp = args.mixed_precision and device.type == "cuda"
    prefetch = (device.type == "cuda") if args.prefetch is None else args.prefetch

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    model_config = dict(export_hf.PUBLISHED_MODEL_CONFIG)
    if args.model_config:
        model_config.update(json.load(open(args.model_config)))
    for k, v in model_config.items():
        setattr(args, k, v)
    assert args.vocab_size == tokenizer.get_vocab_size()
    ws_table = word_start_table(tokenizer)

    train_path = os.path.join(args.data_dir, f"{args.train_dir}_tokenized.bin")
    if not os.path.exists(train_path):
        sys.exit(f"missing {train_path}: run  python prepare_data.py --splits {args.train_dir},dev,train_10M")
    manifest = json.load(open(os.path.join(args.data_dir, f"{args.train_dir}_manifest.json")))
    dev_manifest = json.load(open(os.path.join(args.data_dir, "dev_manifest.json")))
    corpus_tokens = manifest["tokens"]
    if args.smoke:  # slice of the training documents, written next to the run
        docs, keep, n = torch.load(train_path), [], 0
        for d in docs:
            keep.append(d[: args.smoke_train_tokens - n]); n += len(keep[-1])
            if n >= args.smoke_train_tokens:
                break
        train_path = os.path.join(args.out_dir, "smoke_train_slice.bin")
        torch.save(keep, train_path)
        corpus_tokens = n
    if args.max_steps is None:
        args.max_steps = int(round(args.epochs * corpus_tokens / args.batch_tokens))
    print(f"[schedule] {args.train_dir}: {corpus_tokens:,} tokens/epoch -> max_steps {args.max_steps:,} "
          f"({args.max_steps * args.batch_tokens / corpus_tokens:.3f} nominal epochs at {args.batch_tokens} tokens/step)", flush=True)

    m_share = 1 - args.hybrid_numerator / args.hybrid_denominator
    print(f"[hybrid] causal share {args.hybrid_numerator}/{args.hybrid_denominator} = {1 - m_share:.4f}, masked share {m_share:.4f} "
          f"(official train_10m.py equivalent: --hybrid_numerator {args.hybrid_denominator - args.hybrid_numerator} "
          f"--hybrid_denominator {args.hybrid_denominator}, numerator = masked ranks)", flush=True)

    # --- model: seeded exactly like official rank 0 (seed_everything(seed) right before Bert(args))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    model = Bert(args)
    if args.init_from_hf:
        sd = torch.load(os.path.join(args.init_from_hf, "pytorch_model.bin"), map_location="cpu")
        missing, unexpected = model.load_state_dict(export_hf.hf_to_train(sd), strict=False)
        assert not unexpected and all(k.endswith("position_indices") for k in missing)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params:,} parameters; config {model_config}", flush=True)
    model.to(device)

    dev_docs = torch.load(os.path.join(args.data_dir, "dev_tokenized.bin"))
    dev, dev_info = build_dev_subset(args, tokenizer, dev_docs)
    print(f"[dev subset] {dev_info}", flush=True)
    dumper = Dumper(args, tokenizer, args.data_dir, args.out_dir, dev_docs) if args.dumps else None
    if dumper:
        print(f"[dumps] dev {len(dumper.sets['dev']['static']['token_ids']):,} tokens / {dumper.sets['dev']['words']:,} words; "
              f"train sample {len(dumper.sets['train_sample']['static']['token_ids']):,} tokens / {dumper.sets['train_sample']['words']:,} words", flush=True)
    curve = None
    if args.bpw_every:
        if os.path.isdir(args.dev_text_dir) and os.path.exists(DEV_BPW_PY):
            curve = CurveBPW(args)
            print(f"[dev_bpw curve] {[(f, len(i), w) for f, i, w in curve.files]} ctx {args.curve_ctx}", flush=True)
        else:
            print(f"[dev_bpw curve] DISABLED: missing {args.dev_text_dir} or {DEV_BPW_PY}", flush=True)

    # --- run_meta.json: every value explicit
    meta = {"args": vars(args), "explicit_2025_values": {k: getattr(args, k) for k in [
                "vocab_size", "hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers", "layer_norm_eps",
                "learning_rate", "batch_tokens", "seq_length", "seq_switch_fracs", "max_steps", "epochs", "warmup_proportion",
                "cooldown_proportion", "min_lr_factor", "mask_p_start", "mask_p_end", "mask_random_p", "mask_keep_p", "weight_decay",
                "optimizer_eps", "optimizer_beta1", "optimizer_beta2", "max_gradient", "z_loss_weight", "ema_decay",
                "hybrid_numerator", "hybrid_denominator"]} | {"optimizer": "LAMB (official lamb.py)", "batch_reduction": 1,
                "causal_share": 1 - m_share, "masked_share": m_share},
            "repo_head_2024_defaults_NOT_used": REPO_2024_DEFAULTS,
            "official_repo_commit": OFFICIAL_COMMIT, "git": git_info(), "seed": args.seed, "n_params": n_params,
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                         "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                         "numpy": np.__version__, "transformers": __import__("transformers").__version__,
                         "tokenizers": __import__("tokenizers").__version__},
            "host": socket.gethostname(), "device": str(device), "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "tokenizer_sha256": sha256_file(args.tokenizer_path),
            "train_data": {k: manifest[k] for k in ("split", "words_whitespace", "tokens", "token_stream_sha256", "whitespace")},
            "dev_data": {k: dev_manifest[k] for k in ("words_whitespace", "tokens", "token_stream_sha256")},
            "dev_bpw_py_sha256": curve.sha if curve else None,
            "recipe_file_sha256": {f: sha256_file(os.path.join(HERE, f)) for f in
                                   ["train.py", "export_hf.py", "scoring.py", "prepare_data.py", "official/model_extra.py",
                                    "official/dataset.py", "official/lamb.py", "official/utils.py", "hf_files/modeling_gpt_bert.py",
                                    "hf_files/configuration_gpt_bert.py", "hf_files/tokenizer.json"]},
            "dev_subset": dev_info, "resumed_from": args.resume}
    json.dump(meta, open(os.path.join(args.out_dir, "run_meta.json" if not args.resume else "run_meta_resume.json"), "w"), indent=2, default=str)

    if args.eval_only:
        r = evaluate_subset(model, dev, device, args.dev_batch)
        res = {"dev_loss_causal": r["causal"], "dev_loss_masked": r["masked"], "dev_subset": dev_info, "init_from_hf": args.init_from_hf}
        if curve:
            res["dev_bpw_curve"] = curve(model, device)
        if dumper:
            d = dumper.dump(model, "eval_only", device)
            res.update({"dev_dump_loss_causal": d["dev"]["mean_nats"], "dev_dump_bpw": d["dev"]["bpw"],
                        "train_sample_loss_causal": d["train_sample"]["mean_nats"], "train_sample_bpw": d["train_sample"]["bpw"]})
        print(json.dumps(res, indent=2))
        json.dump(res, open(os.path.join(args.out_dir, "eval_only.json"), "w"), indent=2)
        return res

    # --- optimizer exactly as train_10m.py
    no_decay = ['bias', 'layer_norm']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer = Lamb([{'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
                      {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}],
                     args.learning_rate, betas=(args.optimizer_beta1, args.optimizer_beta2), eps=args.optimizer_eps)
    scheduler = cosine_schedule_with_warmup_cooldown(optimizer, int(args.max_steps * args.warmup_proportion),
                                                     int(args.max_steps * args.cooldown_proportion), args.max_steps, args.min_lr_factor)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_params, model_params = list(ema_model.parameters()), list(model.parameters())
    batcher = HybridBatcher(args, tokenizer, train_path, ws_table)

    # --- logging
    csv_path = os.path.join(args.out_dir, "loss_curve.csv")
    cols = ["step", "tokens_seen", "words_seen", "words_seen_est", "epoch", "lr", "seq_len", "train_loss_causal", "train_loss_masked",
            "dev_loss_causal", "dev_loss_masked", "dev_bpw_curve", "dev_dump_loss_causal", "dev_dump_bpw", "train_sample_loss_causal",
            "train_sample_bpw", "elapsed_s", "data_tokens_seen", "grad_norm", "z_term", "mask_p", "checkpoint"]
    start_step, states_dir, hf_dir = 0, os.path.join(args.out_dir, "states"), os.path.join(args.out_dir, "hf")
    os.makedirs(states_dir, exist_ok=True)
    elapsed_offset = 0.0
    if args.resume:
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        for k in ("max_steps", "batch_tokens", "seq_length", "learning_rate", "hybrid_numerator", "hybrid_denominator", "seed", "train_dir"):
            assert st["args"][k] == getattr(args, k), f"resume mismatch on {k}: {st['args'][k]} vs {getattr(args, k)}"
        model.load_state_dict(st["model"]); ema_model.load_state_dict(st["ema_model"])
        optimizer.load_state_dict(st["optimizer"]); scheduler.load_state_dict(st["scheduler"])
        batcher.load_state_dict(st["data"])
        start_step = st["step"]
        torch.set_rng_state(st["rng"]["cpu"]); random.setstate(st["rng"]["python"]); np.random.set_state(st["rng"]["numpy"])
        if device.type == "cuda" and st["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        elapsed_offset = st.get("elapsed_s", 0.0) - (time.time() - t_start)
        old = list(csv.DictReader(open(csv_path))) if os.path.exists(csv_path) else []
        fcsv = open(csv_path, "w", newline="")
        writer = csv.DictWriter(fcsv, fieldnames=cols)
        writer.writeheader()
        for r in old:
            if int(r["step"]) <= start_step:
                writer.writerow(r)
        print(f"[resume] from {args.resume} at step {start_step} ({batcher.words:,} words consumed)", flush=True)
    else:
        fcsv = open(csv_path, "w", newline="")
        writer = csv.DictWriter(fcsv, fieldnames=cols)
        writer.writeheader()

    batches = prefetch_iter(batcher, start_step, args.max_steps, device.type == "cuda") if prefetch else \
        (batcher.batch(s) for s in range(start_step, args.max_steps))
    words_per_token = manifest["words_whitespace"] / manifest["tokens"]

    def row(step, tokens, data_tokens, words, lr, seq_len, acc, sub=None, bpw=None, dump=None, ckpt=""):
        k = max(acc["n"], 1)
        r = {"step": step, "tokens_seen": tokens, "words_seen": words, "words_seen_est": round(data_tokens * words_per_token),
             "epoch": round(data_tokens / corpus_tokens, 5), "lr": lr, "seq_len": seq_len,
             "train_loss_causal": acc["c"] / acc["nc"] if acc["nc"] else "", "train_loss_masked": acc["m"] / acc["nm"] if acc["nm"] else "",
             "dev_loss_causal": sub["causal"] if sub else "", "dev_loss_masked": sub["masked"] if sub else "",
             "dev_bpw_curve": bpw["bits_per_word"] if bpw else "",
             "dev_dump_loss_causal": dump["dev"]["mean_nats"] if dump else "", "dev_dump_bpw": dump["dev"]["bpw"] if dump else "",
             "train_sample_loss_causal": dump["train_sample"]["mean_nats"] if dump else "",
             "train_sample_bpw": dump["train_sample"]["bpw"] if dump else "",
             "elapsed_s": round(time.time() - t_start + elapsed_offset, 2), "data_tokens_seen": data_tokens,
             "grad_norm": acc["g"] / k if acc["n"] else "", "z_term": acc["z"] / k if acc["n"] else "",
             "mask_p": acc["mp"] / k if acc["n"] else "", "checkpoint": ckpt}
        writer.writerow(r); fcsv.flush()
        print(" ".join(f"{c}={r[c]:.4f}" if isinstance(r[c], float) else f"{c}={r[c]}" for c in cols if r[c] != ""), flush=True)

    def new_acc():
        return {"n": 0, "c": 0.0, "nc": 0, "m": 0.0, "nm": 0, "g": 0.0, "z": 0.0, "mp": 0.0}

    def save_state(name, step, data_state):
        with batcher.lock:  # the data thread swaps the global CPU RNG under this lock
            cpu_rng = torch.get_rng_state()
        state = {"step": step, "model": model.state_dict(), "ema_model": ema_model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "data": data_state, "args": vars(args),
                 "rng": {"cpu": cpu_rng, "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                         "python": random.getstate(), "numpy": np.random.get_state()},
                 "elapsed_s": time.time() - t_start + elapsed_offset}
        path = os.path.join(states_dir, f"{name}.pt")
        torch.save(state, path + ".tmp"); os.replace(path + ".tmp", path)
        if args.resume_states == "last":
            for f in os.listdir(states_dir):
                if f.endswith(".pt") and f != f"{name}.pt":
                    os.remove(os.path.join(states_dir, f))

    acc = new_acc()
    if start_step == 0:
        sub0 = evaluate_subset(model, dev, device, args.dev_batch)
        bpw0 = curve(model, device) if curve else None
        row(0, 0, 0, 0, optimizer.param_groups[0]["lr"], seq_len_for_step(0, args), acc, sub0, bpw0)
    model.train()
    t_train, pending, saved = time.time(), [], []
    dump_units = int(round(args.dump_every_words / args.ckpt_word_unit))
    s, b = start_step, None

    for step, b in enumerate(batches, start=start_step):
        ids = b["input_ids"].to(device, non_blocking=True).t()
        tg = b["target_ids"].to(device, non_blocking=True).t()
        am = b["attention_mask"].to(device, non_blocking=True)
        lr = optimizer.param_groups[0]["lr"]
        nm, k = b["num_masked"], args.grad_accum
        nc = ids.size(1) - nm
        assert nm % k == 0 and nc % k == 0, f"--grad_accum {k} must divide masked ({nm}) and causal ({nc}) counts"
        masked_loss = causal_loss = z_loss = None
        for i in range(k):  # micro-batches keep the masked/causal proportion; each micro loss is the official hybrid mix
            if k == 1:
                mi, mt, mm = ids, tg, am
            else:
                sel = torch.cat([torch.arange(i * nm // k, (i + 1) * nm // k), nm + torch.arange(i * nc // k, (i + 1) * nc // k)]).to(device)
                mi, mt, mm = ids[:, sel], tg[:, sel], am[sel]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, ml, cl, _, _, _, zl, _ = model(mi, mm, mt, num_masked=nm // k, ratio=m_share)
            ((loss + args.z_loss_weight * zl) / k).backward()
            if torch.is_tensor(ml):
                masked_loss = ml.detach() / k + (masked_loss if masked_loss is not None else 0)
            if torch.is_tensor(cl):
                causal_loss = cl.detach() / k + (causal_loss if causal_loss is not None else 0)
            z_loss = zl.detach() / k + (z_loss if z_loss is not None else 0)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():  # EMA, as train_10m.py (k = d*k + (1-d)*q)
            torch._foreach_mul_(ema_params, args.ema_decay)
            torch._foreach_add_(ema_params, [p.detach() for p in model_params], alpha=1.0 - args.ema_decay)
        pending.append((causal_loss, masked_loss, grad_norm.detach(), z_loss, b["mask_p"]))
        s = step + 1

        is_final = s == args.max_steps
        ckpts = b["ckpts"]
        is_eval = s % args.eval_every == 0 or bool(ckpts) or is_final
        if s % args.log_every == 0 or is_eval:
            for c, m_, g, z, mp in pending:
                acc["n"] += 1; acc["g"] += g.item(); acc["z"] += z.item(); acc["mp"] += mp
                if c is not None:
                    acc["c"] += c.item(); acc["nc"] += 1
                if m_ is not None:
                    acc["m"] += m_.item(); acc["nm"] += 1
            pending.clear()
            sub = evaluate_subset(model, dev, device, args.dev_batch) if is_eval else None
            bpw = curve(model, device) if curve and (bool(ckpts) or is_final or s % args.bpw_every == 0) else None
            want_dump = dumper is not None and (is_final or any(int(round(w / args.ckpt_word_unit)) % dump_units == 0 for _, w in ckpts))
            dump = dumper.dump(model, "final" if is_final else ckpts[-1][0], device) if want_dump else None
            row(s, b["tokens_total"], b["data_tokens_total"], b["words_total"], lr, b["seq_len"], acc, sub, bpw, dump,
                ";".join(n for n, _ in ckpts) + (";final" if is_final else ""))
            acc = new_acc()
        for name, w in ckpts:
            export_hf.export(model.state_dict(), os.path.join(hf_dir, name), model_config,
                             {"step": s, "words_seen": b["words_total"], "tokens_seen": b["tokens_total"],
                              "data_tokens_seen": b["data_tokens_total"], "threshold_words": w, "weights": "raw (non-EMA)"})
            saved.append({"name": name, "step": s, "words_seen": b["words_total"]})
        if ckpts:
            save_state(ckpts[-1][0], s, b["data_state"])
        if args.stop_after_step is not None and s >= args.stop_after_step:
            print(f"[stop] --stop_after_step {s}", flush=True)
            fcsv.close()
            return {"stopped_at": s}

    train_time = time.time() - t_train
    fcsv.close()
    # --- final artefacts (raw weights are the primary export; EMA kept alongside)
    final_sub = evaluate_subset(model, dev, device, args.dev_batch)
    ema_sub = evaluate_subset(ema_model, dev, device, args.dev_batch)
    info = {"step": s, "words_seen": batcher.words, "tokens_seen": batcher.tokens, "data_tokens_seen": batcher.data_tokens}
    export_hf.export(model.state_dict(), os.path.join(hf_dir, "final"), model_config, dict(info, weights="raw (non-EMA)"))
    export_hf.export(ema_model.state_dict(), os.path.join(hf_dir, "final_ema"), model_config, dict(info, weights=f"EMA {args.ema_decay}"))
    save_state("final", s, b["data_state"])
    png = None
    try:
        out = subprocess.run([sys.executable, os.path.join(HERE, "..", "..", "tools", "plot_curve.py"), csv_path],
                             capture_output=True, text=True, timeout=300)
        png = out.stdout.strip() or None
        if out.returncode:
            print(out.stderr)
    except Exception as e:  # plotting must never kill a finished run
        print(f"[plot] failed: {e}")
    fd = os.path.join(args.out_dir, "dumps", "final", "summary.json")
    final_dump = json.load(open(fd)) if dumper and os.path.exists(fd) else None
    summary = {"n_params": n_params, "device": str(device), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
               "train_time_s_this_process": train_time, "total_time_s": time.time() - t_start + elapsed_offset,
               "steps_per_s_this_process": (s - start_step) / max(train_time, 1e-9), "final": info, "epochs": batcher.data_tokens / corpus_tokens,
               "dev_subset_final_raw": final_sub, "dev_subset_final_ema": ema_sub,
               "dev_dump_final": {k: final_dump["dev"][k] for k in ("mean_nats", "bpw", "tokens", "words")} if final_dump else None,
               "train_sample_final": {k: final_dump["train_sample"][k] for k in ("mean_nats", "bpw", "tokens", "words")} if final_dump else None,
               "checkpoints": saved, "loss_curve_csv": csv_path, "loss_curve_png": png}
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2, default=str)
    meta["summary"] = summary
    json.dump(meta, open(os.path.join(args.out_dir, "run_meta.json" if not args.resume else "run_meta_resume.json"), "w"), indent=2, default=str)
    print(f"[done] {s} steps; words consumed {batcher.words:,} ({batcher.data_tokens / corpus_tokens:.3f} epochs); "
          f"dev subset raw {final_sub} EMA {ema_sub}; dev dump {summary['dev_dump_final']}", flush=True)
    return summary


if __name__ == "__main__":
    main()
