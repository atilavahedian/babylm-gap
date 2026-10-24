# Reading task: our scores are ~0 because of the target-word token form (11 Oct)

The experimenter asked for a sanity check. Reading_rt and Reading_spr are 0.00-0.29 for every one of our models. The certified published GPT-BERT is at 9.5-9.9 and 3.3-3.5.

| model | Reading_rt | Reading_spr |
|---|---|---|
| pub_gptbert_causal (certified) | 9.89 | 3.45 |
| pub_gptbert_mixed (certified) | 9.50 | 3.37 |
| base_S_e6, S_e6_s1/s2, T100_S_e10 | 0.14-0.19 | 0.00-0.08 |
| L28_e6_s0-s2 | 0.13-0.19 | 0.00-0.01 |
| mix28L_p01/p23/p45 | 0.17-0.29 | 0.00-0.02 |

## Cause

The cause is verified from code and outputs.

The official `evaluation_pipeline/reading/evaluation_functions.py:get_p2` (bf55c1131e53654c2a87418f5629c26959acd710) scores the target word as `tokenizer(word, add_special_tokens=False)["input_ids"]`, with no leading space.

Our export uses the GPT-2 byte-level BPE with `add_prefix_space=False`. So `the` encodes to the word-internal token `the`, not `Ġthe`, and the same holds for `cup` and `watching`. The task therefore scores P(word-internal piece | context). That is why `prediction.jsonl` shows 10.3 nats for "the" after "Arthur placed" (GPT-BERT: 2.25). The surprisal predictor then has no relation to reading times (RTfirstfix coefficient -0.10, p=0.64).

GPT-BERT's tokenizer normalizer prepends a space to every input, so it gets `Ġthe`.

## Scope

Only Reading is affected. The other 6 composite tasks and Entity (sentence_zero_shot/dataset.py) tokenize the whole sentence and locate the completion by character offsets, so word boundaries are correct there.

Effect on comparisons:
- Arm comparisons between our own models are not biased, because both arms carry the same ~0. But 2 of the 8 composite tasks carry no signal.
- Absolute composites are about 1.6 points (≈13/8) below what a correctly tokenized reading score would give. So they are **not comparable to GPT-BERT or the leaderboard**.

## Alignment check (critic 81702719, condition i)

Script: `babylm/eval/reading_alignment_check.py`. Stimuli: full_eval/reading/reading_data.csv (sha256 ref-0391…), 3247 targets (1726 words + 1521 previous words). Raw results are in `babylm/results/reading_align/`.

| tokenizer | critic definition: tok(word) ids = in-sentence ids | pipeline-effective: ids get_p2 scores = in-sentence ids |
|---|---|---|
| our export as is | 3247 mismatch (100%) | 3247 (100%) |
| our export, add_prefix_space=True | **0 (0.00%)** | 269 (8.28%) |
| published GPT-BERT (as is) | 0 (0.00%) | 308 (9.49%) |

- By the critic's definition, the prefix-space copy has 0% mismatch, so it PROCEEDS.
- The pipeline-effective mismatches come from official get_p2 itself. It re-encodes every continuation piece of a multi-piece word through the tokenizer, so a word-final "." or "," becomes "Ġ." / "Ġ,". GPT-BERT has the same quirk at a slightly higher rate (9.49%). So the prefix-space copy reproduces the leaderboard's own behaviour, which is what the amended column is for.

Critic ruling (11 Oct): PROCEED acknowledged. The get_p2 quirk stays unpatched. Report footnote for the amended column: "Amended after data (critic 81702719): reading re-scored with the export's tokenizer set to add_prefix_space=True, weights unchanged; 0.00% target misalignment; the official get_p2 re-encodes continuation pieces of multi-piece words on 8.28% of targets for our tokenizer vs 9.49% for published GPT-BERT. Never changes a verdict."
