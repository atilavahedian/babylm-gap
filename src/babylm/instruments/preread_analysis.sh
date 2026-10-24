#!/bin/bash
# Phase 2 pre-read analyses on the scores from preread_queue.sh -> results/babylm/preread_v1/ (CSV + PNG).
# Each step skips pairs or models whose scores are not there yet, so it can be rerun as the queue fills.
cd "$(dirname "$0")/../.." || exit 1
T=${1:-preread_v1}
DATA_ROOT=${BABYLM_DATA:-~/babylm_data}
python3 -m babylm.instruments.units --data-root "$DATA_ROOT"     # document ids: bootstrap and cross-fit unit
python3 -m babylm.instruments.overlap --data-root "$DATA_ROOT"   # words in lines verbatim in train_10M / train_100M
python3 -m babylm.instruments.slices --tag "$T" \
  --pair gpt2_10m/chck_100M:gpt2_100m/chck_1000M \
  --pair gpt2_10m/chck_100M:gpt2_100m/chck_100M \
  --pair gpt2_10m/chck_10M:gpt2_10m/chck_100M \
  --pair gptbert_cf_10m/chck_100M:gptbert_cf_100m/chck_1000M \
  --pair gpt2_10m/chck_100M:gptbert_cf_10m/chck_100M \
  --pair gptbert_cf_10m_collapse/chck_100M:gptbert_cf_100m_collapse/chck_1000M \
  --pair gpt2_10m_tabspace/chck_100M:gpt2_100m_tabspace/chck_1000M \
  --pair gpt2_10m_tabkeep/chck_100M:gpt2_100m_tabkeep/chck_1000M \
  --ref gptbert_cf_100m/chck_1000M,gpt2_100m/chck_1000M
python3 -m babylm.instruments.bottleneck --tag "$T" --ten gpt2_10m --hundred gpt2_100m
# GPT-BERT 100M differs in size (120M vs 31M), batch (8x) and schedule: finals only, no equal-exposure claims
python3 -m babylm.instruments.bottleneck --tag "$T" --ten gptbert_cf_10m --hundred gptbert_cf_100m --no-equal-exposure
python3 -m babylm.instruments.anatomy --tag "$T" \
  --cache gpt2_10m/chck_100M,gpt2_100m/chck_100M,gpt2_100m/chck_1000M,gptbert_cf_10m/chck_100M,gptbert_cf_100m/chck_1000M \
  --confidence gpt2_10m/chck_100M,gpt2_100m/chck_100M,gpt2_100m/chck_1000M,gptbert_cf_10m/chck_100M,gptbert_cf_100m/chck_1000M \
  --temps gpt2_10m/chck_100M,gpt2_100m/chck_1000M,gptbert_cf_10m/chck_100M,gptbert_cf_100m/chck_1000M \
  --mix gpt2_10m/chck_100M+gptbert_cf_10m/chck_100M \
  --mix gpt2_10m/chck_50M+gpt2_10m/chck_100M \
  --mix gpt2_10m/chck_70M+gpt2_10m/chck_80M+gpt2_10m/chck_90M+gpt2_10m/chck_100M \
  --mix gptbert_cf_10m/chck_50M+gptbert_cf_10m/chck_100M \
  --mix gptbert_cf_10m/chck_70M+gptbert_cf_10m/chck_100M
# Kneser-Ney count models on train_10M (legal) and train_100M (oracle), mixed into each model
python3 -m babylm.instruments.ngram --tag "$T" --data-root "$DATA_ROOT" \
  --models gpt2_10m/chck_100M,gpt2_100m/chck_100M,gpt2_100m/chck_1000M,gptbert_cf_10m/chck_100M,gptbert_cf_100m/chck_1000M \
  --joint gptbert_cf_10m/chck_100M+kn:train_10M+cache \
  --joint gpt2_10m/chck_100M+kn:train_10M+cache \
  --joint gptbert_cf_10m/chck_100M+gpt2_10m/chck_100M+kn:train_10M+cache
# where learning stalls (GPT-BERT trained on 128-token sequences for its first 60%: compare within those positions)
python3 -m babylm.instruments.stall --tag "$T" --run gpt2_10m --ref gpt2_100m/chck_1000M
python3 -m babylm.instruments.stall --tag "$T" --run gptbert_cf_10m --ref gptbert_cf_100m/chck_1000M --max-tok-pos 128
python3 -m babylm.instruments.signature --tag "$T"
python3 -m babylm.instruments.dispersion --tag "$T" --data-root "$DATA_ROOT"
python3 -m babylm.instruments.prize --tag "$T" --data-root "$DATA_ROOT"
