#!/usr/bin/env bash
# One AMLM training run on a GPU VM with the authors' pristine pinned code. Run from a checkout of this repo.
#   usage: vm_job.sh <arm> <seed> <out_dir>
#   arms: hard_decay_off | hard_off    official BabyLM 2025 train_10M (reference arm, 3 seeds); tokenizer retrained on
#                                      train_10M only with the authors' sentencepiece spec
#         hard_decay_pub | hard_pub    authors' bb24.train + tokenizer (fidelity label only, critic ref-0088)
# Data: official zips from OSF (md5-checked), or pre-staged at $OFFICIAL_ZIPS/{train_10M,dev}.zip. Test is never read.
# Output in <out_dir>: train.log, curve.{csv,png}, mlm_probs.csv, checkpoint-* (BabyLM 1M..10M and 10M..100M word
# points), per_token/ (MLM pseudo-loss on full DEV and a 1M-word train sample at each epoch checkpoint), run.json.
# PT_DEV_WORDS caps the DEV words scored per checkpoint (default 0 = all). PT_STAGE=dir defers per-token scoring:
# its inputs are copied there and pt_epochs.sh is run later (gpu_job.sh does this after the official eval).
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; OUT="${3:?out_dir}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"; WORK="$OUT/work"; mkdir -p "$WORK"
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || python3 -m pip install -q torch
python3 -m pip install -q "transformers==4.54.1" "datasets>=2.18,<4" sentencepiece protobuf "bitsandbytes>=0.43" matplotlib
bash "$HERE/fetch.sh" "$WORK/code"

# Official text (needed by every arm: DEV is the official dev split).
for z in train_10M:y7djq:dc67e5d2db96ed4f76cb072e2504a18c dev:uyd7v:ad712e2f943ddbbbc94036178144e810; do
  IFS=: read -r name id md5 <<<"$z"
  if [ -n "${OFFICIAL_ZIPS:-}" ]; then cp "$OFFICIAL_ZIPS/$name.zip" "$WORK/"; else curl -sSL --retry 5 -o "$WORK/$name.zip" "https://osf.io/download/$id/"; fi
  echo "$md5  $WORK/$name.zip" | md5sum -c - ; python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$WORK/$name.zip" "$WORK/text"
done
python3 "$HERE/prep_official.py" "$WORK/text/train_10M" "$WORK/text/dev" "$WORK/off" > "$OUT/data_manifest.json"

EXTRA=(--lamb)
if [ -n "${AMLM_CPU_TEST:-}" ]; then  # CI-style check of this script on CPU: 100 lines, no LAMB, device patched
  sed -i -e 's/"cuda:0"/"cpu"/g' -e 's/^from bitsandbytes.optim import LAMB/LAMB = None/' \
         -e 's/torch.cuda.memory_allocated()/0/g; s/torch.cuda.memory_reserved()/0/g' "$WORK/code/train_mask.py"
  EXTRA=(--debug --batch_size 8)
fi
cd "$WORK/code"
case "$ARM" in
  *_pub) TRAIN=data/bb24.train; DEV_SMALL=data/bb25_small.dev; TOK=tokenizers/bb24.model ;;
  *_off) TRAIN="$WORK/off/train_10M.txt"; DEV_SMALL="$WORK/off/dev_small.txt"; TOK="$OUT/tok_off.model"
         # The authors' trainer_spec (read from tokenizers/bb24.model), trained on official train_10M only.
         python3 - "$TRAIN" "${TOK%.model}" <<'PY'
import sys, sentencepiece as spm
spm.SentencePieceTrainer.train(input=sys.argv[1], model_prefix=sys.argv[2], model_type="bpe", vocab_size=40000,
    character_coverage=0.9995, byte_fallback=True, user_defined_symbols=["[MASK]"], max_sentence_length=4192,
    shuffle_input_sentence=True, split_digits=False, num_threads=16, seed_sentencepiece_size=1000000,
    pad_id=0, bos_id=1, eos_id=2, unk_id=3, pad_piece="[PAD]", bos_piece="[CLS]", eos_piece="[SEP]",
    unk_piece="[UNK]", unk_surface=" ⁇ ")
PY
         ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
[ -n "${AMLM_DRYRUN:-}" ] && { echo "dry run: data + tokenizer ok ($TRAIN, $TOK)"; exit 0; }
case "$ARM" in hard_decay_*) MASK=(--mlm_prob 0.4 --mask_decay 0.25) ;; hard_*) MASK=(--mlm_prob 0.15) ;; esac
START=$(date -u +%s)
python3 train_mask.py --train_data "$TRAIN" --valid_data "$DEV_SMALL" --tokenizer "$TOK" --output_path "$OUT" \
  --mask_update_steps 200 --logging_steps 200 --intermediate_size 1280 --hidden_size 384 --max_seq_len 0:64,5:256 \
  --all_checkpoints --log_mlm_probs --seed "$SEED" --cpus "$(nproc)" "${MASK[@]}" "${EXTRA[@]}" 2>&1 | tee "$OUT/train.log"
END=$(date -u +%s)
python3 "$HERE/../curves.py" "$OUT/train.log" "$OUT/curve" --epochs 10 --title "AMLM $ARM seed $SEED"

# Words seen per checkpoint (whitespace words; 10 epochs over the training file), and per-token losses at epochs.
TOTAL=$(grep -m1 -oP 'Total steps: \K\d+' "$OUT/train.log")
TRAIN_WORDS=$(python3 -c "import sys;print(sum(len(l.split()) for l in open(sys.argv[1],encoding='utf-8')))" "$TRAIN")
python3 "$HERE/add_fast_tok.py" "$OUT"/checkpoint-* > /dev/null  # eval venv needs tokenizer.json
for c in $(ls -d "$OUT"/checkpoint-* | awk -F"checkpoint-" '{print $NF" "$0}' | sort -n | cut -d" " -f2-); do
  s=${c##*-}; echo "$s $(python3 -c "print(round($s/$TOTAL*10*$TRAIN_WORDS))")" >> "$OUT/words_seen.txt"
done
if [ -n "${PT_STAGE:-}" ]; then  # defer per-token scoring (gpu_job.sh runs the official eval first)
  mkdir -p "$PT_STAGE"; cp -r "$WORK/text/dev" "$WORK/off/train_sample_1M.txt" "$PT_STAGE/"; cp "$TOK" "$PT_STAGE/tok.model"
else
  bash "$HERE/pt_epochs.sh" "$OUT" "$TOK" "$WORK/text/dev" "$WORK/off/train_sample_1M.txt"
fi
python3 - "$OUT/run.json" "$ARM" "$SEED" "$START" "$END" "$TRAIN" "$TOK" "$TRAIN_WORDS" "$TOTAL" "$HERE" <<'PY'
import json, sys, hashlib, subprocess, torch
o, arm, seed, s, e, train, tok, words, total, here = sys.argv[1:]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(dict(arm=arm, seed=int(seed), train_wall_s=int(e) - int(s), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               train_sha256=h(train), train_words=int(words), words_exposed=10 * int(words), total_steps=int(total),
               tokenizer_sha256=h(tok), torch=torch.__version__,
               upstream_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               hinton_commit=subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               deviations_from_paper="none from code; code differs from paper: LAMB betas (0.9,0.98) not 0.95, z-loss 1e-4, bf16 weights"),
          open(o, "w"), indent=1)
PY
rm -rf "$WORK"
echo "done: $OUT"
