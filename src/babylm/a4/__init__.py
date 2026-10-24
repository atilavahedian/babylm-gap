"""Attack A4, closed-system re-expression (docs/babylm/attacks/A4_reexpression.md): the base writes new contexts for
familiar words from train_10M seeds, and a student continues from it on synthetic + real text inside the R0 budget.

ngram     token/word n-gram hashing, the copy-block index, overlap and leakage primitives
kvcache   KV-cache decoding and sampling for the sprint-1 GPT family (sizes S and M)
data      document halves, seed tiling, synthetic-split writer, N2 sentence shuffle
generate  CLI: one shard of synthetic text from a generator checkpoint (seeded or self-seeded, copy-blocked or not)
audit     CLI: 13-gram leakage vs eval sets and DEV, n-gram overlap and nearest-neighbour edit distance vs train_10M
ledger    CLI: per-arm exposure and FLOP ledger from run outputs (R0, critic amendments 2 and 3)
verdict   CLI: the screen verdict by the registered kill rule (audits first; no verdict before the floor)
The student's second LR cycle lives in the harness (babylm.harness.cycle, train.py --init_from / --sources).
"""

# Clean-tree note: only t2.py (imported by babylm.harness.train) lives in common/; the rest of babylm.a4 lives in
# experiments/2026-10-09-babylm-attacks/code/babylm/a4/ and is found through extend_path when that folder is on PYTHONPATH.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
