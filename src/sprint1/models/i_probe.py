"""Pre-registered one-exposure fact-learning probe (OXF) for agent I (card: sprint1/ideas/I_kwta.md).

Not a model: build() is absent on purpose. Deterministic probe set (seed 2026) of 512 nonce facts
    "The <attr> of <Name> is <Value>."
split 256 studied / 256 unstudied. The harness inserts each studied fact once into training (as its own
<|endoftext|>-delimited document at the steps given by `insertion_steps`); `oxf_score` measures
    OXF = mean_studied log p(Value | prefix) - mean_unstudied log p(Value | prefix)   [nats per fact]
with a bootstrap CI over facts.  Each per-fact score is PAIRED to remove value difficulty:
    score(f) = log p(Value | "The attr of Name is") - log p(Value | "The attr of Foil is")
where Foil is a nonce name that never appears in training; OXF = mean_studied score - mean_unstudied score. Token ids come from any callable `encode(str) -> list[int]` (GPT-2 BPE via
tiktoken with TIKTOKEN_CACHE_DIR=sprint1/measure/data/tiktoken_cache).
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

ATTRS = ["capital", "owner", "river", "mascot", "anthem", "currency", "patron", "harbor",
         "language", "festival", "mountain", "emblem", "dynasty", "oracle", "guild", "harvest"]
CONS, VOW = "bdfgklmnprstvz", "aeiou"
EOT = 50256


def _word(rng: random.Random, lo: int = 2, hi: int = 4) -> str:
    return "".join(rng.choice(CONS) + rng.choice(VOW) for _ in range(rng.randint(lo, hi))).capitalize()


def make_probe_set(seed: int = 2026, n: int = 512) -> list[dict]:
    rng = random.Random(seed)
    names, facts = set(), []
    while len(facts) < n:
        name = _word(rng)
        if name in names:
            continue
        names.add(name)
        facts.append({"attr": rng.choice(ATTRS), "name": name, "value": _word(rng)})
    for f in facts:  # foil names: never inserted anywhere, disjoint from every probe name
        while True:
            foil = _word(rng)
            if foil not in names:
                names.add(foil)
                f["foil"] = foil
                break
    order = list(range(n))
    rng.shuffle(order)
    for r, i in enumerate(order):
        facts[i]["studied"] = r < n // 2
    return facts


def fact_tokens(fact: dict, encode, foil: bool = False) -> tuple[list[int], list[int]]:
    prefix = encode(f"The {fact['attr']} of {fact['foil' if foil else 'name']} is")
    value = encode(f" {fact['value']}.")
    return prefix, value


def fact_document(fact: dict, encode) -> list[int]:
    """The training document inserted once for a studied fact."""
    p, v = fact_tokens(fact, encode)
    return [EOT] + p + v


def insertion_steps(total_steps: int, n_studied: int = 256) -> list[int]:
    """Evenly spaced steps in [40%, 60%] of training, one studied fact per listed step."""
    lo, hi = 0.4 * total_steps, 0.6 * total_steps
    return [int(lo + (hi - lo) * i / max(1, n_studied - 1)) for i in range(n_studied)]


@torch.no_grad()
def value_logprob(model, prefix: list[int], value: list[int], device="cpu") -> float:
    ids = torch.tensor([[EOT] + prefix + value], device=device)
    x, y = ids[:, :-1], ids[:, 1:]
    logits = model.logits(model.hidden(x)) if hasattr(model, "hidden") else None
    if logits is None:
        raise ValueError("model must expose hidden()/logits() (split API)")
    lp = -F.cross_entropy(logits[0].float(), y[0], reduction="none")
    return float(lp[-len(value):].sum())


def oxf_score(model, encode, facts=None, n_boot: int = 2000, seed: int = 0, device="cpu") -> dict:
    facts = facts or make_probe_set()
    model.eval()
    s, u = [], []
    for f in facts:
        lp = value_logprob(model, *fact_tokens(f, encode), device=device)
        lp_foil = value_logprob(model, *fact_tokens(f, encode, foil=True), device=device)
        (s if f["studied"] else u).append(lp - lp_foil)
    st, ut = torch.tensor(s), torch.tensor(u)
    g = torch.Generator().manual_seed(seed)
    boots = [(st[torch.randint(len(st), (len(st),), generator=g)].mean()
              - ut[torch.randint(len(ut), (len(ut),), generator=g)].mean()).item() for _ in range(n_boot)]
    boots.sort()
    return {"oxf": float(st.mean() - ut.mean()), "ci95": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]),
            "studied_paired_mean": float(st.mean()), "unstudied_paired_mean": float(ut.mean()), "n": len(facts)}
