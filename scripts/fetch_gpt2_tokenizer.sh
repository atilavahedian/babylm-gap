#!/usr/bin/env bash
# Restore the public GPT-2 tokenizer files used by the GPT-2 baseline recipe and the sprint-1 port.
# They are not vendored here; the copies the experiments used are byte-identical to Hugging Face's openai-community/gpt2.
set -euo pipefail
cd "$(dirname "$0")/.."
U=https://huggingface.co/openai-community/gpt2/resolve/main
for d in src/babylm/recipes/sprint1_port/gpt2_tokenizer src/babylm/recipes/gpt2/base; do
  for f in vocab.json merges.txt; do curl -sSL -o "$d/$f" "$U/$f"; done
done
curl -sSL -o src/babylm/recipes/sprint1_port/gpt2_tokenizer/tokenizer.json "$U/tokenizer.json"
echo "GPT-2 tokenizer files restored"
