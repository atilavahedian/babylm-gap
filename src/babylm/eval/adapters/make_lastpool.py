"""python make_lastpool.py SRC_HF_DIR DST_DIR : copy a GPT-2 HF dir and register modeling_lastpool as its AutoModel."""
import json, os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
shutil.copytree(src, dst, dirs_exist_ok=True)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modeling_lastpool.py'), dst)
c = json.load(open(os.path.join(dst, 'config.json')))
assert c.get('model_type') == 'gpt2', c.get('model_type')
c['auto_map'] = {'AutoModel': 'modeling_lastpool.GPT2LastTokenPoolModel'}
json.dump(c, open(os.path.join(dst, 'config.json'), 'w'), indent=2)
print('lastpool adapter ->', dst)
