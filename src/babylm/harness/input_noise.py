"""Attack A2: input-noise augmentation (babylm.harness.train --noise_mode / --noise_r2l). KNOWN FAMILY, not new:
training-time token augmentation for data-constrained AR pretraining (Chen et al., arXiv 2606.16246: uniform random
replacement 15% is their best single method, -0.17 nats at the minimum; R2L 50% with a direction token -0.10),
masked-input regularisation (Xu et al., arXiv 2606.06888, as an auxiliary loss = a second forward), word dropout
(Gal & Ghahramani 2016), SwitchOut (Wang et al. 2018), data noising as smoothing (Xie et al. 2017). Card:
docs/babylm/attacks/A2_input_noise.md.

Off by default (--noise_mode none, --noise_r2l 0): train.py builds nothing and the port's step is unchanged. On:
every causal row of a micro-batch is transformed on the CPU right before it goes to the device, on the SAME forward
pass, and its targets keep the clean tokens (the label sequence is never modified), so every window still counts
once against the exposure cap (critic R0). MNTP rows (--mntp_frac) are never touched. Eval is always clean.

  replace  each eligible INPUT position is replaced with probability p by a token drawn from --noise_dist:
           uniform over the real vocabulary minus BOS, or the train_10M unigram distribution (BOS excluded).
  mask     the same positions get the MASK id (16384) instead: information removed, no distractor (null twin).
  gauss    no token changes: the raw input embedding of every eligible position gets isotropic Gaussian noise of
           the same expected squared size as a replacement at rate p, sigma_x^2 = p * E_r ||e_r - e_x||^2 / d with r
           drawn from --noise_dist, computed from the live embedding table each forward (null twin). It is a forward
           hook on model.embed, registered before torch.compile, active only in training mode on causal rows.
  --noise_r2l q  each causal row is reversed with probability q (direction token R2L = 16385 at position 0): a row
           with n real targets y[:n] becomes y' = flip(y[:n]), x' = [R2L, y'[0], ..., y'[n-2]], padding unchanged.
           The target positions are exactly the L2R row's (same tokens, same words), so exposure is unchanged.
           L2R rows carry no direction token (the off path stays the port's input format).

Eligible positions: real input tokens (y >= 0 at that position) that are not BOS (document separator) and not the
R2L marker. Replacement tokens never include BOS, MASK or R2L. mask and R2L need ids >= 16384, so train.py pads the
model vocab to 16512 for them (as MNTP does; a padded vocab changes the init RNG stream, so those arms are not
init-matched to an unpadded base: the seed floor covers that, and the screen's base arm is run at the same vocab
whenever the base itself uses MNTP).
Corruption draws come from a private CPU generator seeded from the data seed (state in resume.pt): the global RNG,
the model init and the window order are untouched. gauss draws its noise from the device RNG inside the forward.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MODES = ("none", "replace", "mask", "gauss")
DISTS = ("uniform", "unigram")
MASK_ID = 16384  # = levers' MNTP MASK (train.py mask_id)
R2L_ID = 16385
SEED_OFFSET = 9_000_011


def needs_pad(mode: str, r2l: float) -> bool:
    return mode == "mask" or r2l > 0


def replacement_probs(dist: str, vocab_real: int, protect_ids=(0,), counts: np.ndarray | None = None) -> np.ndarray:
    """float64 [vocab_real] distribution of replacement tokens: uniform or unigram, protected ids at 0."""
    if dist == "uniform":
        w = np.ones(vocab_real, dtype=np.float64)
    elif dist == "unigram":
        if counts is None:
            raise ValueError("unigram replacement needs token counts")
        w = np.asarray(counts[:vocab_real], dtype=np.float64).copy()
    else:
        raise ValueError(f"unknown noise dist {dist!r}")
    w[list(protect_ids)] = 0.0
    if w.sum() <= 0:
        raise ValueError("replacement distribution is empty")
    return w / w.sum()


class InputNoise:
    def __init__(self, mode: str, p: float, dist: str = "uniform", r2l: float = 0.0, *, vocab_real: int = 16384,
                 protect_ids=(0,), counts: np.ndarray | None = None, seed: int = 0,
                 mask_id: int = MASK_ID, r2l_id: int = R2L_ID):
        if mode not in MODES:
            raise ValueError(f"unknown noise mode {mode!r}")
        if not 0.0 <= p < 1.0 or not 0.0 <= r2l <= 1.0:
            raise ValueError(f"need 0 <= noise_p < 1 and 0 <= noise_r2l <= 1 (got {p}, {r2l})")
        if mode == "none" and p > 0:
            raise ValueError("--noise_p needs a --noise_mode")
        self.mode, self.p, self.dist, self.r2l = mode, float(p), dist, float(r2l)
        self.protect = torch.tensor(sorted(set(protect_ids)), dtype=torch.long)
        self.mask_id, self.r2l_id = int(mask_id), int(r2l_id)
        self.probs = replacement_probs(dist, vocab_real, protect_ids, counts)
        self._probs_t = torch.from_numpy(self.probs)
        self._allowed = torch.from_numpy(np.flatnonzero(self.probs > 0))
        self.gen = torch.Generator().manual_seed(int(seed) + SEED_OFFSET)
        self.last_r2l: torch.Tensor | None = None  # bool [B] rows reversed by the last call (None: no R2L draw)
        self.reset_stats()

    # -------------------------------------------------------------------------------------------- stats
    def reset_stats(self) -> None:
        self.hook_on = True  # gauss: the training loop turns it off around MNTP rows
        self.n_elig, self.n_corrupt, self.n_rows, self.n_r2l = 0, 0, 0, 0

    def pop_stats(self) -> dict[str, float]:
        out = {"noise_frac": self.n_corrupt / max(1, self.n_elig), "noise_r2l_frac": self.n_r2l / max(1, self.n_rows)}
        self.reset_stats()
        return out

    # -------------------------------------------------------------------------------------------- token side
    def eligible(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (y >= 0) & ~torch.isin(x, self.protect) & (x != self.r2l_id)

    def sample_tokens(self, n: int) -> torch.Tensor:
        if self.dist == "uniform":
            return self._allowed[torch.randint(0, len(self._allowed), (n,), generator=self.gen)]
        return torch.multinomial(self._probs_t, n, replacement=True, generator=self.gen)

    def reverse_rows(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        rows = torch.rand(x.shape[0], generator=self.gen) < self.r2l
        self.last_r2l = rows  # babylm.harness.stack: A1 neither reads nor writes its cache on reversed rows
        if not rows.any():
            return x, y, 0
        x, y = x.clone(), y.clone()
        for r in torch.nonzero(rows).flatten().tolist():
            n = int((y[r] >= 0).sum())
            if n == 0:
                continue
            if not bool((y[r, :n] >= 0).all()):
                raise ValueError("R2L needs the real targets of a row to be a prefix")
            yr = y[r, :n].flip(0)
            y[r, :n] = yr
            x[r, 0] = self.r2l_id
            x[r, 1:n] = yr[:n - 1]
        return x, y, int(rows.sum())

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x, y: CPU int64 [B, T] causal rows. Returns (noisy inputs, targets) with the same target multiset."""
        self.n_rows += x.shape[0]
        self.last_r2l = None
        if self.r2l > 0:
            x, y, nr = self.reverse_rows(x, y)
            self.n_r2l += nr
        if self.mode in ("replace", "mask") and self.p > 0:
            elig = self.eligible(x, y)
            hit = (torch.rand(x.shape, generator=self.gen) < self.p) & elig
            k = int(hit.sum())
            self.n_elig += int(elig.sum())
            self.n_corrupt += k
            if k:
                x = x.clone()
                x[hit] = self.mask_id if self.mode == "mask" else self.sample_tokens(k)
        return x, y

    # -------------------------------------------------------------------------------------------- gauss null
    def attach_gauss(self, embed: nn.Embedding) -> None:
        """Energy-matched Gaussian input-embedding noise (mode gauss): forward hook on the token embedding."""
        V = len(self.probs)
        w = torch.from_numpy(self.probs).float()
        protect = self.protect

        def hook(mod: nn.Module, inp, out: torch.Tensor):
            if not (mod.training and self.hook_on and self.p > 0):
                return out
            ids = inp[0]
            W = mod.weight[:V].detach().float()
            wd = w.to(W.device)
            ebar = wd @ W                          # E_r e_r            [d]
            msq = wd @ (W * W).sum(1)              # E_r ||e_r||^2      []
            e = out.detach().float()
            d2 = (e * e).sum(-1) + msq - 2.0 * (e @ ebar)  # E_r ||e_r - e_x||^2  [B, T]
            sigma = (self.p * d2.clamp_min(0.0) / e.shape[-1]).sqrt()
            sigma = sigma * (~torch.isin(ids, protect.to(ids.device))).to(sigma.dtype)
            return out + (torch.randn_like(e) * sigma.unsqueeze(-1)).to(out.dtype)

        self.hook_handle = embed.register_forward_hook(hook)

    # -------------------------------------------------------------------------------------------- resume
    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state(), "mode": self.mode, "p": self.p, "dist": self.dist, "r2l": self.r2l}

    def load_state_dict(self, d: dict) -> None:
        if (d["mode"], d["p"], d["dist"], d["r2l"]) != (self.mode, self.p, self.dist, self.r2l):
            raise ValueError("noise state is for different --noise_* settings")
        self.gen.set_state(d["gen"])
