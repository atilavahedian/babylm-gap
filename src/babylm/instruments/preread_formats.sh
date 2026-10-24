#!/bin/bash
# Format sensitivity on the published finals (critics ref-0257, ref-0490, ref-0259; ref-0056).
# GPT-2 speaker-label tabs: the per-source mapping in preread_queue.sh (CHILDES tabs -> space, Switchboard tabs kept)
# was chosen after per-source train-text scores, so the GPT-2 headline is the best uniform setting on the probe (all
# tabs -> space, runs gpt2_*_tabspace); all tabs kept (the code is silent on tabs) is the sensitivity row (_tabkeep).
# --reuse copies every chunk whose text the variant leaves unchanged from the run scored in preread_queue.sh (chunks
# are scored independently) and scores only the rest: 2% of words for tabspace, 30% for tabkeep, 55% for collapse.
# GPT-BERT whitespace: the GPT-BERT 2025 reference collapses whitespace inside lines (babylm/data/corpus.py
# normalize_line on the experimenter's branch) and the train-text probe agrees, so its finals are also scored with --collapse
# (gptbert_cf_*_collapse); both prizes are shown. Headlines first, the tabkeep sensitivity last.
cd "$(dirname "$0")/../.." || exit 1
P="python3 -m babylm.instruments.score --splits dev,train"
G10="--repo BabyLM-community/babylm-baseline-10m-gpt2 --line-join newline_space --revisions chck_100M --reuse gpt2_10m"
G100="--repo BabyLM-community/babylm-baseline-100m-gpt2 --line-join newline_space --revisions chck_1000M --reuse gpt2_100m"
GB="--line-join newline --tab-space childes,switchboard --collapse"
$P --run gpt2_10m_tabspace $G10 --tab-space childes,switchboard
$P --run gpt2_100m_tabspace $G100 --tab-space childes,switchboard
$P --run gptbert_cf_10m_collapse --repo BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus $GB --revisions chck_100M --reuse gptbert_cf_10m
$P --run gptbert_cf_100m_collapse --repo BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus $GB --revisions chck_1000M --reuse gptbert_cf_100m
$P --run gpt2_10m_tabkeep $G10
$P --run gpt2_100m_tabkeep $G100
echo FORMATS_DONE
