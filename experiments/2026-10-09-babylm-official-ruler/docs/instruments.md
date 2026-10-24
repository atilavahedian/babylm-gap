# BabyLM bottleneck instruments (Phase 2)

Reusable: every attack is read on the same frozen samples and slices.

## Samples (frozen before any score; critic ruling 2026-10-09T18:34Z)

- `babylm/instruments/samples/dev_sample_v1.index.tsv`: 419 chunks, 138,745 words of the official 2025 **dev** split (file, line range, sha256 per chunk). Dropped any chunk with more than 5% of its 13-grams in train_100M; sample 13-gram overlap is 0.05% vs train_100M and 0.0015% vs train_10M.
- `babylm/instruments/samples/train_sample_v1.index.tsv`: 280 chunks of **train_10M** (a subset of train_100M), for memorisation slices only, labelled train.
- `babylm/instruments/samples/manifest_v1.json`: zip and file sha256s, seed, parameters, sample and feature hashes.
- Source shares by words differ from the full dev split (simple_wiki 7% here vs 13% in dev, because its chunks are shorter). Per-source slices are reported separately; the sample is for slicing, not for the official held-out number.

## Unit

Causal loss summed over each whitespace word (nats/word or bits/word), so different tokenizers compare fairly. A token belongs to the word it starts in; a whitespace token belongs to the next word. 512-token windows with a BOS token; longer chunks continue with stride 256.

**Text format must match training.** Two traps, each worth 1 to 4 nats/word on the affected text:

- Line joins. The GPT-2 baselines were trained on `' '.join(f.readlines())` (a space after every newline, momergul/babylm-gpt2-baseline `data_utils.py`); scored on plain newlines they lose about 1.55 nats/word to a mispredicted lone-space token at every line start. GPT-BERT was trained on plain newlines (0.8 to 1.1 nats/word worse if given the extra space).
- Tabs after speaker labels (CHILDES `*CHI:\t`, Switchboard `A:\t`). The tokenizer maps the tab differently per pipeline: GPT-2 saw CHILDES tabs as spaces but kept Switchboard tabs; GPT-BERT saw both as spaces. Wrong choice costs 1.4 to 4.4 nats/word on those sources.

`format_probe.py` picks both from loss on a fixed subset of the **train** sample only (never dev). Train-text nats/word for the finals:

| model | newline / newline_space | CHILDES tab keep / space | Switchboard tab keep / space | args |
|---|---|---|---|---|
| GPT-2 10M (epoch 1) | 7.85 / 6.64 | 6.88 / 4.95 | 5.47 / 6.46 | `--line-join newline_space --tab-space childes` |
| GPT-2 100M final | 6.19 / 4.33 | 5.48 / 2.92 | 3.56 / 5.18 | `--line-join newline_space --tab-space childes` |
| GPT-BERT causal 10M final | 4.05 / 5.19 | 7.82 / 2.93 | 6.62 / 3.64 | `--line-join newline --tab-space childes,switchboard` |
| GPT-BERT causal 100M final | 3.87 / 4.96 | 7.16 / 2.79 | 6.14 / 3.38 | `--line-join newline --tab-space childes,switchboard` |

`score.py` requires `--line-join` and records both choices in `meta.json`; the words, and so the denominator, are the same in every format. Any new model (a reproduced winner, an attack) is probed first and scored in its own training format; `preread_queue.sh` is the exact pre-read recipe.

## Slices

- `freq10` / `freq100`: count of the word's normal form (lowercase, outer punctuation stripped) in train_10M / train_100M.
- `position`: word index in the chunk (how much context the model has).
- `ngram10`: longest n-gram (n <= 8) ending at the word that occurs in train_10M; 0 = word never seen, 1 = word seen but never after this previous word.
- `context`: first occurrence in the chunk, word repeated, or bigram repeated (in-context copying).
- `source`: the six corpus sources.
- Memorisation: train-sample vs dev loss by frequency band (train frequency excludes the word's own occurrence) and greedy span recall (8/16/32 tokens all argmax-correct, one teacher-forced pass).

## Probes

Every probe reads the stored per-word scores, writes CSV + PNG to `results/babylm/<tag>/`, and skips models not scored yet. Weights or temperatures are always fitted on even documents and scored on odd documents, and vice versa; CIs are 95% document-cluster bootstraps (`units.py`: documents as in the experimenter's `babylm/data/corpus.py`; 300 documents in the 419-chunk dev sample, since long books span several chunks; chunk clusters gave CIs 3 to 8% narrower).

- `slices.py --pair a:b`: the gap between two checkpoints per slice, on dev and on the train sample (text both models trained on, so the train gap is "seen text" and dev minus train is "new text only"). `--ref` adds a reference-difficulty slice from a third model. Slices include `word_kind` (speaker label, punctuation only, function word from a fixed closed-class list, content word) `pieces` (visible tokens per word in model a's own scoring tokenization; most multi-piece words are a word plus attached punctuation) and `token_class` (the gap summed over each model's own whitespace, punctuation and letter tokens).
- `units.py`: document id per sample word, the unit for every bootstrap and cross-fit.
- `overlap.py`: flags dev-sample words in lines (8+ words) found verbatim in train_10M or train_100M, the rule in the experimenter's `p2_slices.py` (critic ref-0260: bottlenecks are sized with train_100M overlap removed); `prize.py` reports the gap without them.
- `bottleneck.py`: per epoch of a 10M run vs its 100M twin: what each epoch buys vs 10x data per slice, loss by times-seen (exposure), repeated vs fresh text at equal words seen, and each epoch in fresh-word equivalents. `--no-equal-exposure` for pairs that differ in size or schedule (GPT-BERT).
- `stall.py`: where learning stalls. For every checkpoint: dev and train-sample loss per frequency group, share of the epoch-1-to-reference distance closed, late gain per epoch with CI and epochs to the reference at that rate, and the generalisation gap (dev minus train). `--max-tok-pos 128` for GPT-BERT, whose first 60% of training used 128-token sequences.
- `signature.py`: lever signatures. For any weaker:stronger pair (10x data, more epochs, fresh vs repeated text, a recipe change, an attack vs its base): share of loss removed per slice, plus a rare tilt (words seen 1 to 15 vs 64+ times) and a novelty tilt (trigram in neither corpus vs in train_10M). Extra epochs sit at 0.43 to 0.76 on novelty; 10x data sits at 1.04 and the GPT-2-to-GPT-BERT recipe at 0.96.
- `anatomy.py`: `--mix` mixture of two or more models (token level within one run, word level across tokenizers): bounds what ensembling or ensemble distillation could buy; `--confidence` loss minus the model's own entropy per frequency band (above 0 = overconfident); `--temps` one global softmax temperature (needs `score.py --temps`); `--cache` a copy cache of earlier words in the chunk.
- `ngram.py`: interpolated Kneser-Ney word trigrams on train_10M (a legal add-on) and train_100M (an oracle for local word statistics), mixed into each model. `--joint` fits one mixture of several components (models, count models, the cache) and reports the overlap with their separate gains.
- `dispersion.py`: contexts per type. Within each frequency band, first occurrences split into terciles of dispersion (range over 1000-word blocks of train_10M vs the range expected at that count) and of left diversity; gap per tercile.
- `prize.py`: one figure of the 10M-to-100M gap: seen vs new text (and the new-text part split into words seen under 64 and 64+ times in train_10M, per band dev share x (dev minus train gap per word)), frequency groups, the count-model oracle, the cheap legal add-ons; `prize_bands.csv` has the band shares with document CIs. `--data-root` adds the total reweighted to the official dev split's source word shares and the gap with speaker labels (`*CHI:`, `A:`) left out (also a `word_kind` slice in `slices.py`). It also reports the gap and the bands without words carrying CHILDES conversation-analysis markup (`slices.CHAT_MARKUP`: ⌊ ⌋ ⌈ ⌉ ↗ ∆ ∙ ⁎), which train_10M never contains, so no 10M-only method can learn them: 0.10% of the dev sample's words carry 0.026 of GPT-BERT's 0.846 (0.820 without them).
- `rare_split.py`: bottleneck 2 (words seen under 64 times in train_10M) split by first occurrence in the chunk vs a repeat, and by whether the 10M tokenizer cuts the word into a rare piece (token count under 64 in train_10M), with the prize_bands protocol; also each model's own loss on these words by the rarity of its own pieces (the two tokenizers differ).
- `export_dumps.py`: per-token arrays (nats, word index, chunk, position, token id, train_10M token count) and a per-word CSV (word, normal form, train_10M/100M counts, first in chunk, overlap flag, each model's nats) for other threads.
- `harness.py`: reads the experimenter's harness dumps (`pw/pw_{dev,train}_<label>.npy`, float16 nats per whitespace word of DEV-sub and train-sub, about 1M words each, one per saved epoch). `features` rebuilds both subsets as one row per document (`samples/harness_subsets/`, copied from the experimenter's ref-0396) with the same word features; `import` copies a run's dumps to `data/babylm_scores/<run>/chck_<label>/{devsub,trainsub}.npz`. Every probe then reads them with `BABYLM_SAMPLE=harness` (dev means DEV-sub, train means train-sub). Rows are whole documents, so the bootstrap unit is the document. Token-level probes (window positions, token mixtures, token classes) are not available on dumps. The harness scores one stream: the BOS before each document is counted in its first word, and early words of a document see the previous document as context (port final, words 1 to 50 of a document: 11.5 nats in the dump vs 9.7 scored per document; words 200 and later track closely, r = 0.99 and means within 2%).
- `cells.py`: many harness 10M runs (a recipe sweep) against a 100M reference and base runs (the port, the sweep's plain control), on DEV-sub with document CIs. Per run: loss at each dumped epoch; the gap to the reference at the best epoch and the final; the gain and the rare and novelty tilts of epochs 1 to 2, 2 to 5 and 5 to 10 (does it still overfit, and on which words); and the same for what it buys over each base, at the finals and at the best epochs. Gaps are also given without CHILDES markup words (`gap_best_nomarkup`).
- `bands.py`: one harness 10M run's DEV-sub loss per train_10M frequency band (8k+, 64-8k, 1-63, never seen, and never seen without CHILDES markup) at every dumped epoch, its gap to a 100M reference per band, and the change per band between chosen epochs (`--change 70M:100M`), with document CIs.
- `headcheck.py`: runs harness HF exports on DEV-sub documents and reports, per band of the target token's train_10M count, loss minus the model's own entropy (above 0 = overconfident; document CI), the target logit, the share of targets at the softcap floor, the max logit and the output-row norm. On 60 documents the port's final puts 39% of targets seen 1 to 15 times under -14 (twin 0%) and is overconfident by 4.6 nats/token overall (twin 0.36). Rows of earlier exports stay in `headcheck.csv` (same `--docs`), so exports can be added as they land.
- `scorers.py`: splits the registered DEV metric (dev_bpw.py) minus the harness full-DEV loss into six protocol steps (blank lines kept as text, file as one document, BOS at window start, 512/256 windows vs 1024-token blocks, fp32 vs bf16) on whole sampled DEV documents, per source and model, with the separator tokens broken out.
- `babylm/eval/dev_bpw_twin.py` (experimenter 04:22Z, critic ref-0020): the registered metric with only the blank-line handling changed to match the training stream: each run of blank lines becomes one scored BOS and each document is tokenized on its own; it imports dev_bpw.py (same flags) and runs it unchanged, so it needs that file beside it. Switchboard (no blank lines) scores exactly as registered. Second number `ex_unseen_chars_nats_per_word` (critic ref-0015, the ruler's ex_markup rule): drops every word holding one of the 85 never-seen DEV characters (`babylm/tools/unseen_chars_list.json` @ ref-0016, brought in unchanged) and the tokens whose text lies in it; whitespace tokens and separators stay. Its per-token nll_of reproduces dev_bpw's total to the bit (test against both scorer versions). From a v2 dump, use the same token rule: a per-word mask also drops the newline after each markup line. The exported model must load correctly: under transformers 5.19 the remote-code export leaves its rotary buffers unset (NaN losses), so check a known score first.
- `devfull_docs.py`: two groups of harness runs (e.g. solo vs packed seeds) compared on full DEV from their final dumps (`pw_devfull_final` per word, or `pt_devfull_final` per token summed into words), per source, per CHILDES corpus and per document; words holding a character never seen in train_10M (CHILDES conversation-analysis markup) are split out, and "markup-free DEV" drops the documents with 20 or more of them.
- `rowprobe.py`: where HF exports put their rare and never-seen vocabulary rows (tied embedding norm, capped and pre-softcap logit, log-probability on a fixed train text) for ids never in the train_10M stream, the CHILDES conversation-analysis markup bytes and ids seen over 100 times; built to compare packed and solo runs.
- `harness_battery.sh TAG RUN10 REF100 [FRESH]`: slices, stall, signatures, cache, mixtures, count models and the prize for one harness 10M run against its 100M twin, at the final checkpoint and, when an earlier epoch is best on DEV-sub, there too.

## Commands

```
python -m babylm.instruments.prepare --data-root <dir with zips/ and unzipped dev, train_10M, train_100M>
python -m babylm.instruments.format_probe --local-path <HF-format checkpoint dir>     # prints --line-join / --tab-space
python -m babylm.instruments.score --run gpt2_10m --repo BabyLM-community/babylm-baseline-10m-gpt2 --line-join newline_space --tab-space childes --revisions chck_10M,chck_100M
python -m babylm.instruments.score --run my_attack --local-path <HF-format checkpoint dir> --line-join newline --tab-space childes,switchboard
python -m babylm.instruments.score --run port_S_10M_e10_s0 --local-path <harness export dir> --line-join newline --collapse --dtype fp32   # format from babylm/recipes/harness/format.json; model code must be in HINTON_CODE
python -m babylm.instruments.slices --tag preread_v1 --pair gpt2_10m/chck_100M:gpt2_100m/chck_1000M
python -m babylm.instruments.signature --tag preread_v1 --lever "my attack=base_run/chck_100M:my_attack/chck_100M"
bash babylm/instruments/preread_queue.sh && bash babylm/instruments/preread_analysis.sh   # the whole pre-read
bash babylm/instruments/preread_formats.sh   # format sensitivity: GPT-2 finals with one tab rule (headline) and all tabs kept; GPT-BERT finals with whitespace collapse; --reuse rescoring only the chunks a format changes
python -m babylm.instruments.harness features --data-root <data dir>   # once
python -m babylm.instruments.harness import --pack shared/babylm/p2/results_pack/<run>
bash babylm/instruments/harness_battery.sh harness_port port_S_10M_e10_s0 grid_S_100M_e10_s0/chck_1000M
```

GPT-2 runs in bf16 (per-word difference from fp32: mean 0.0003, mean absolute 0.016 nats). GPT-BERT runs in fp32 with an exact re-implementation of its relative-position attention (`gptbert_fast.py`, checked bit-equal to the published code at load), 3x faster on CPU. The published GPT-BERT modeling file is vendored unchanged (sha256 in `vendor_gptbert/__init__.py`); no remote code runs.
