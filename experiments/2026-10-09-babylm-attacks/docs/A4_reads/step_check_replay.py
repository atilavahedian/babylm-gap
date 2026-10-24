"""A4 step check (critic, experimenter 2026-10-10 14:48Z; the K3 bug): replay the pinned trainer's samplers on CPU and report,
for every A4 run, the step at which its last word milestone fires against its final step (a score is VOID unless the
scored checkpoint is the final step). Real data is exact; synthetic splits do not exist before the run, so for mixed
cycles the real part is exact and the synthetic words that keep the end on a milestone are printed.

    git archive ref-0048 babylm sprint1 | tar -x -C PIN    # A4 batch 1/2 pin (T2 pins: same samplers, milestones)
    python -I step_check_replay.py PIN DATA                # DATA/bin/{train_10M,train_1M}.* (babylm.data.prepare)
"""
import math
import sys

import numpy as np

pin, data = sys.argv[1], sys.argv[2]
sys.path.insert(0, pin)
from babylm.data import corpus as C  # noqa: E402
from babylm.harness import cycle as CY  # noqa: E402
from babylm.harness.stream import MILESTONES, EpochSampler, words_in_targets  # noqa: E402

T, BS, END = 1024, 64, 0.99  # seq_len, batch_tokens / seq_len, train.py END_FIRE_FRAC


def plain(split, epochs):
    """Cumulative words after each step of an --epochs run (EpochSampler, data_seed 0, words_in_targets)."""
    d = C.load_split(split, data)
    wi = C.word_initial(np.asarray(d.words)).astype(np.int64)
    smp = EpochSampler(d.n_tokens, T, 0)
    steps = math.ceil(epochs * d.n_tokens / (BS * T))
    return np.cumsum([words_in_targets(wi, smp.next_starts(BS), T) for _ in range(steps)])


def cycle(specs):
    """Cumulative words after each step of a --sources cycle (CycleSampler, data_seed 0, its windows' w_words)."""
    cyc = CY.CycleSampler([CY.load_source(x, data) for x in specs], CY.parse_window(None, T), BS * T, 0)
    w = np.asarray(cyc.w_words[:cyc.total_steps * cyc.batch_seqs], dtype=np.int64)
    return np.cumsum(w.reshape(cyc.total_steps, -1).sum(1))


def report(name, cum, prior=0):
    """train.py's firing: in-loop at the first step reaching each milestone above the prior; at the end the first
    pending one if words >= 99% of it; else a "final" row (no checkpoint at the pinned commits)."""
    cum = cum + prior
    fired = {m: int(np.argmax(cum >= m)) + 1 for m in MILESTONES if prior < m <= cum[-1]}
    pending = [m for m in MILESTONES if m > cum[-1]]
    n, end = len(cum), int(cum[-1])
    if pending and end >= END * pending[0]:
        last = (pending[0], n, "end-fire")
    else:
        m = max(fired)
        last = (m, fired[m], "in-loop")
    ok = last[1] == n
    print(f"{name:22s} steps {n:5d} words {end:>12,}  last checkpoint chck_{last[0] // 10**6}M at step {last[1]} "
          f"({last[2]})  {'OK: the final step' if ok else 'VOID: a final row follows'}")
    return end


def mixed(name, prior, real_specs, target):
    """Real part exact; the end milestone's window for the synthetic part's counted words."""
    r = int(cycle(real_specs)[-1])
    lo = prior + r
    m = next(x for x in MILESTONES if x > lo + target * 0.9)
    print(f"{name:22s} prior {prior:,} + real {r:,}: with S synthetic words the run ends at {lo:,} + S; target S = "
          f"{target:,}; the end milestone {m // 10**6}M fires on the final step for S in [{END * m - lo:,.0f}, "
          f"{m - lo:,}) (end-fire) or a crossing inside the last step (~41k words)")


if __name__ == "__main__":
    e4 = report("check: E4 (= a4_G)", plain("train_10M", 4))  # v2_S_10M_e4_s0 result.json: 39,816,934 words, 974
    e5 = report("check: E5 (= a4_t2_G)", plain("train_10M", 5))  # v2_S_10M_e5_s0: 49,791,035 words, 1218 steps
    report("a4_n0a", cycle(["real:train_10M:6"]), e4)
    report("a4_n0b", cycle(["real:train_10M:4"]), e4)
    mixed("a4_a, a4_n1a, a4_n2a", e4, ["real:train_10M:3"], 29_861_319)
    mixed("a4_b, a4_n1b, a4_n2b", e4, ["real:train_10M:2"], 19_907_546)
    report("a4_t2_n0", cycle(["real:train_10M:5"]), e5)
    mixed("a4_t2, a4_t2_n1, _n2", e5, ["real:train_10M:5"], 100_000_000)
    g1 = report("a4_t2g_G", plain("train_1M", 5))
    report("a4_t2g_base, a4_t2g_M", plain("train_1M", 10))
    report("a4_t2g_n0", cycle(["real:train_1M:5"]), g1)
    mixed("a4_t2g, a4_t2g_n1, _n2", g1, ["real:train_1M:5"], 10_000_000)
