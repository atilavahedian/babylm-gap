"""Instrument the official pipeline (ref-0243) to log argmax ties without changing any score (critic rule 3).
compute_results.py picks torch.max(...)[1], so on a tie the first candidate wins. After each batch's argmax we append
one JSON line {task, data_path, temp, n, ties, nonfinite} to $BABYLM_TIE_LOG (if set). Idempotent.
Usage: python patch_ties.py EVAL2025_DIR"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'evaluation_pipeline/sentence_zero_shot/compute_results.py'
s = p.read_text()
MARK = '# [hinton tie log]'
if MARK not in s:
    old = '        chosen_sentences = torch.max(stacked_probs, dim=1)[1].tolist()\n'
    assert s.count(old) == 2, 'unexpected pipeline version'
    new = old + (f'        {MARK}\n'
                 '        import os as _os, json as _json\n'
                 '        if _os.environ.get("BABYLM_TIE_LOG"):\n'
                 '            _mx = stacked_probs.max(dim=1, keepdim=True)[0]\n'
                 '            _ties = int(((stacked_probs == _mx).sum(dim=1) > 1).sum())\n'
                 '            _nf = int((~torch.isfinite(stacked_probs)).any(dim=1).sum())\n'
                 '            with open(_os.environ["BABYLM_TIE_LOG"], "a") as _f:\n'
                 '                _f.write(_json.dumps({"task": args.task, "data_path": str(args.data_path), "temp": float(temp), '
                 '"n": int(stacked_probs.shape[0]), "ties": _ties, "nonfinite": _nf}) + "\\n")\n')
    s = s.replace(old, new)
    p.write_text(s)
print('patched', p)
