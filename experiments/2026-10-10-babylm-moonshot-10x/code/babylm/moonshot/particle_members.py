"""Split a particle-ensemble checkpoint (babylm/models/particles.py) into its K members, each a plain baseline
checkpoint, so the registered scorer (export_hf + dev_bpw.py, or score_trial.sh) scores members exactly as any model:

    python -m babylm.moonshot.particle_members RUN/checkpoints/chck_100M.pt --out DIR     # DIR/member_<k>.pt

Member k = the shared embedding / blocks 0..Ls-1 / head plus particle k's private blocks and skip weights: the same
function as particle k inside the ensemble. For card PE's J and rho: J = mean member DEV - p_bar DEV, which is
`ceiling_report c3` over the K member dumps (its K = n row; the p_bar DEV equals the ensemble's own registered DEV).
Members are diagnostics only (each is a K-th of a jointly trained network, never a submission).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def member_state(sd: dict[str, torch.Tensor], extra: dict, k: int) -> dict[str, torch.Tensor]:
    K, Ls = int(extra.get("pe_k", 1)), int(extra.get("pe_shared", 4))
    if not 0 <= k < K:
        raise ValueError(f"member {k} of {K}")
    out = {n: t for n, t in sd.items() if not n.startswith(("pblocks.", "pskip."))}
    if k == 0 or extra.get("pe_tied", False):
        return out
    for n, t in sd.items():
        if n.startswith(f"pblocks.{k - 1}."):
            j, rest = n.split(".", 3)[2:]
            out[f"blocks.{Ls + int(j)}.{rest}"] = t
    out["skip_weights"] = sd[f"pskip.{k - 1}"]
    assert set(out) == {n for n in sd if not n.startswith(("pblocks.", "pskip."))}
    return out


def split_checkpoint(ckpt: Path, out: Path) -> list[Path]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("model_name") != "particles":
        raise SystemExit(f"{ckpt}: not a particle checkpoint ({ck.get('model_name')!r})")
    cfg = dict(ck["cfg"])
    extra = dict(cfg.get("extra") or {})
    K = int(extra.get("pe_k", 1))
    base_extra = {n: v for n, v in extra.items() if not n.startswith("pe_")}
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(K):
        m = {n: v for n, v in ck.items() if n not in ("model", "optimizer", "ema")}
        m.update(model=member_state(ck["model"], extra, k), model_name="baseline",
                 cfg={**cfg, "extra": base_extra}, label=f"{ck.get('label')} member {k}/{K} (diagnostic)",
                 particle_source=str(ckpt), particle_member=k)
        p = out / f"member_{k}.pt"
        torch.save(m, p)
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for p in split_checkpoint(Path(a.ckpt), Path(a.out)):
        print(p)


if __name__ == "__main__":
    main()
