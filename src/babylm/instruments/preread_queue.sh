#!/bin/bash
# Phase 2 pre-read: score the published BabyLM 2025 baselines on the frozen samples.
# Formats were inferred per model from train-text loss only (format_probe.py):
#   GPT-2:    lines joined with "\n " (trained on ' '.join(readlines())), CHILDES tabs -> space, Switchboard tabs kept
#   GPT-BERT: plain "\n", CHILDES and Switchboard tabs -> space
cd "$(dirname "$0")/../.." || exit 1
P="python3 -m babylm.instruments.score"
GPT2_FMT="--line-join newline_space --tab-space childes"
GB_FMT="--line-join newline --tab-space childes,switchboard"
G10="--run gpt2_10m --repo BabyLM-community/babylm-baseline-10m-gpt2 $GPT2_FMT"
G100="--run gpt2_100m --repo BabyLM-community/babylm-baseline-100m-gpt2 $GPT2_FMT"
B10="--run gptbert_cf_10m --repo BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus $GB_FMT"
B100="--run gptbert_cf_100m --repo BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus $GB_FMT"
# A: core contrasts (10M final, 100M final, 100M at equal words seen, epoch 1)
$P $G10 --splits dev,train --revisions chck_100M
$P $G100 --splits dev,train --revisions chck_1000M,chck_100M
$P $G10 --splits dev,train --revisions chck_10M
$P $B10 --splits dev,train --revisions chck_100M
$P $G100 --splits dev --revisions chck_10M
echo "PHASE_A_DONE"
# B: the strong recipe at 100M (GPT-BERT 100M is 120M params vs 31M at 10M, and 8x the batch,
# so only its final is a clean reference), temperature checks on the 10M finals, then the epoch curves
$P $B100 --splits dev,train --revisions chck_1000M
TEMPS="--splits dev --temps 0.8,0.9,1.1,1.2,1.3 --cache data/babylm_ckpt_tmp_t"
$P $G10 --revisions chck_100M $TEMPS
$P $B10 --revisions chck_100M $TEMPS
$P $G100 --revisions chck_1000M $TEMPS
$P $B10 --splits dev,train --revisions chck_10M,chck_20M,chck_50M
$P $G10 --splits dev,train --revisions chck_20M,chck_30M,chck_50M,chck_70M
# GPT-BERT epoch 7 first: it opens the late-epoch window (7 to 10) for the stronger family
$P $B10 --splits dev,train --revisions chck_70M
$P $G100 --splits dev --revisions chck_20M,chck_50M
$P $B10 --splits dev --revisions chck_30M,chck_80M,chck_90M
$P $B100 --splits dev --revisions chck_100M
$P $B100 --revisions chck_1000M $TEMPS
echo "PHASE_B_DONE"
# C: fill-ins
$P $G10 --splits dev --revisions chck_1M,chck_2M,chck_3M,chck_5M,chck_40M,chck_60M,chck_80M,chck_90M
$P $B10 --splits dev --revisions chck_1M,chck_2M,chck_5M
$P $G100 --splits dev --revisions chck_1M,chck_2M,chck_5M,chck_200M,chck_300M,chck_500M,chck_700M
$P $B100 --splits dev --revisions chck_300M
echo "PHASE_C_DONE"
