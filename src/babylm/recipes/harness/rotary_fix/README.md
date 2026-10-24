# Rotary buffers lost on load under newer transformers (experimenter's harness export)

Patch for `babylm/harness/hf_template/modeling_hinton.py` (experimenter's branch, @ ref-0252). Apply on that branch with
`git apply babylm/recipes/harness/rotary_fix/modeling_hinton_rotary.patch`, then re-export.

**Bug.** `Rotary` keeps `freq`, `cos` and `sin` as non-persistent buffers, so they are not in `model.safetensors`.
Transformers 5.19 builds the model on the meta device and never materialises them, and `_init_weights` is a no-op.
So `cos[0,0]` is garbage, and every logit and loss is NaN, for both AutoModelForCausalLM and AutoModel.
Transformers 4.51.3, the version pinned by the certified official pipeline, loads the model correctly.

**Fix.** `Rotary` keeps `head_dim`, `base` and `frac` as plain attributes. It rebuilds the tables on the first forward
after loading, computing them on CPU in fp32 as at export time and then moving them. `_init_weights` also marks the
tables stale. Tables come out bit-identical, so exports scored under 4.51.3 are unaffected.

**Test** (`test_rotary_reload.py`: tiny random model built with each template; loss and logit sum after
AutoModelForCausalLM load):

| template | transformers 4.51.3 | transformers 5.19.0 |
|---|---|---|
| old | loss 5.585857391357422, logit sum 427.67474212139314 | **NaN** |
| patched | loss 5.585857391357422, logit sum 427.67474212139314 | loss 5.585857391357422, logit sum 427.67474212139314 |

AutoModel last_hidden_state sum: old 87.35888 / NaN; patched 87.35888 / 87.35888.

`babylm/recipes/sprint1_port/hf_template/modeling_sprint1.py` caches its tables in a Python dict (no buffers), so it
is not affected.

The ruler's `vm_job.sh` also refuses to score under any transformers other than the pinned 4.51.3. It does not score
(it writes VOID_<name>) any model whose smoke test shows non-finite causal logits or that fails to load.
