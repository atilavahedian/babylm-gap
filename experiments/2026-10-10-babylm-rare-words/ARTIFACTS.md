# Artifacts not stored in this folder

A5 trained no model, so no checkpoints exist. The diagnostic `a5_diag` was ticketed but never launched: its planned
outputs (`shared/babylm/attacks/A5/<name>/`) do not exist, and the executor ledger has no admit or close for it.

| artifact | where | size |
|---|---|---|
| Unredacted rare-word tables (per-word "Top 1-63 words by gap mass" lists; BabyLM corpus words) | wip-branch:results/babylm/rare_words_v1/decomp.md, decomp_filtered.md, decomp_filtered.json | 6,083 / 6,136 / 25,528 B |
| Unredacted full-dev anatomy (560 example words from the dev split, by never-seen category) | wip-branch:results/babylm/rare_words_v1/rare_words_fulldev_anatomy.json | 25,269 B |
| Harness with the `--focus_plan` wiring (full file; this folder has only the patch vs ref-0426) | wip-branch:babylm/harness/train.py | 52,856 B |
| GPT-BERT baselines used for the anatomy | Hugging Face: `babylm-baseline-10m-gpt-bert-causal-focus`, `babylm-baseline-100m-gpt-bert-causal-focus` (public, BabyLM 2025) | not recorded |
| BabyLM train_10M / train_100M / dev raw text (inputs; never committed) | OSF download via `python -m babylm.data.prepare --download` | not recorded |
| Base run whose per-word dumps produced `a5_bands_fulldev/` (`base_S_wd0p4_lr0p5_r1_ve0_tied`, `pw_devfull_final.npy`) | shared/babylm/base_r1/ (executor 3 VM copy-back; see ../2026-10-09-babylm-gap-map/ARTIFACTS.md for the base runs' GCS paths) | not recorded |
| Rare-word analysis scripts | common/babylm/analysis/rare_words/ (shared code, not duplicated here) | - |
