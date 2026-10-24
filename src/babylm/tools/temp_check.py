"""CPU diagnostic: free post-hoc temperature calibration on a DEV sample. Diagnostic only, never a verdict.
Scores 1024-token windows spread evenly over data/babylm/bin/dev.bin (positions 256..1023, so every scored token has
>= 256 tokens of context) and saves, per model, the target's softcapped logit and logsumexp(z / T) on a T grid.
usage: python3 temp_check.py OUT.npz NWIN [--vocab_mask V] hf_dir1 hf_dir2 ...   (read with temp_read.py)
--vocab_mask V sets logits >= V to -inf (exports padded past the 16,384-token tokenizer, e.g. A2's 16,512)."""
import sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

args = sys.argv[1:]
vmask = 0
if '--vocab_mask' in args:
    i = args.index('--vocab_mask')
    vmask = int(args[i + 1])
    del args[i:i + 2]
out, nwin, dirs = args[0], int(args[1]), args[2:]
dev = np.fromfile('data/babylm/bin/dev.bin', dtype=np.uint16)
L, P0 = 1024, 256
starts = np.linspace(0, dev.size - L - 1, nwin).astype(np.int64)
Ts = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5], dtype=np.float32)
torch.set_num_threads(4)
res = {'starts': starts, 'Ts': Ts, 'vocab_mask': np.int64(vmask)}
for mi, d in enumerate(dirs):
    m = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=torch.float32).eval()
    zy, lse = [], []
    t0 = time.time()
    with torch.inference_mode():
        for wi, s in enumerate(starts):
            x = torch.from_numpy(dev[s:s + L].astype(np.int64))[None]
            z = m(x).logits[0, P0 - 1:L - 1].float()  # predicts tokens P0..L-1
            if vmask:
                z[:, vmask:] = float('-inf')
            y = x[0, P0:L]
            zy.append(z.gather(1, y[:, None])[:, 0].numpy())
            lse.append(torch.stack([torch.logsumexp(z / T, 1) for T in Ts.tolist()], 1).numpy())
            if wi % 32 == 0:
                print(f'model {mi} win {wi}/{nwin} {time.time() - t0:.0f}s', flush=True)
    res[f'zy{mi}'] = np.concatenate(zy)
    res[f'lse{mi}'] = np.concatenate(lse)
res['y'] = np.concatenate([dev[s + P0:s + L] for s in starts]).astype(np.int64)
np.savez(out, **res)
print('saved', out)
