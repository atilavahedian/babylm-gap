"""Per-word DEV losses split by words containing a character never present in train_10M raw text.
usage: python3 -I unseen_chars.py NAME=file.npy ...   (pt_devfull_final.npy per token, or pw_devfull_final.npy per word)"""
import sys, json, numpy as np, os
B='data/babylm/'
R=B+'raw/'
tr=set()
for f in os.listdir(R+'train_10M'): tr|=set(open(R+'train_10M/'+f,encoding='utf-8').read())
dv=set()
for f in os.listdir(R+'dev'): dv|=set(open(R+'dev/'+f,encoding='utf-8').read())
never=dv-tr
print(f'DEV chars {len(dv)}, never in train_10M: {len(never)}', ''.join(sorted(never))[:120])
tj=json.load(open('shared/babylm/base_r1/base_S_wd0p4_lr0p5_r1_ve0_tied/hf/tokenizer.json'))
vocab=tj['model']['vocab']
# GPT-2 byte<->unicode map
bs=list(range(ord('!'),ord('~')+1))+list(range(ord('¡'),ord('¬')+1))+list(range(ord('®'),ord('ÿ')+1)); cs=bs[:]; n=0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+n); n+=1
u2b={chr(c):b for b,c in zip(bs,cs)}
V=max(vocab.values())+1
tb=[b'']*V
for s,i in vocab.items():
    try: tb[i]=bytes(u2b[ch] for ch in s)
    except KeyError: tb[i]=s.encode()
for a in tj.get('added_tokens',[]): tb[a['id']]=b''
nonascii=np.array([any(x>=128 for x in t) for t in tb])
dev=np.fromfile(B+'bin/dev.bin',dtype=np.uint16); wid=np.fromfile(B+'bin/dev.words.bin',dtype=np.uint32)
meta=json.load(open(B+'bin/dev.meta.json')); W=meta['words']
cand=np.unique(wid[nonascii[dev]])
# token index ranges per word (word ids are nondecreasing along the stream)
starts=np.searchsorted(wid,cand,side='left'); ends=np.searchsorted(wid,cand,side='right')
bad=np.zeros(int(wid.max())+1,bool)
for w,a,b in zip(cand,starts,ends):
    s=b''.join(tb[t] for t in dev[a:b]).decode('utf-8',errors='replace')
    if any(ch in never for ch in s): bad[w]=True
print(f'words with a never-seen char: {int(bad.sum())} ({bad.sum()/W:.3%} of {W} words)')
src=np.fromfile(B+'bin/dev.src.bin',dtype=np.uint8)
wsrc=np.zeros(int(wid.max())+1,np.uint8); wsrc[wid]=src
srcs=list(meta['per_source'])
res={}
for arg in sys.argv[1:]:
    name,p=arg.split('=',1); x=np.load(p).astype(np.float64); x[np.isnan(x)]=0
    pw=np.bincount(wid,weights=x,minlength=bad.size) if x.size==dev.size else x
    tot=pw.sum()/W; ub=pw[bad[:pw.size]].sum()/W
    per={s:pw[(wsrc[:pw.size]==i)&bad[:pw.size]].sum()/W for i,s in enumerate(srcs)}
    res[name]=(tot,ub,per)
    print(f'{name:10s} total {tot:.4f}  never-seen-char words {ub:.4f}  rest {tot-ub:.4f}   (by source: '+', '.join(f'{s[:5]} {v:.4f}' for s,v in per.items())+')')
ns=list(res)
for i in range(1,len(ns)):
    b,n=ns[0],ns[i]
    print(f'{b} - {n}: total {res[b][0]-res[n][0]:+.4f}, from never-seen-char words {res[b][1]-res[n][1]:+.4f}, rest {(res[b][0]-res[b][1])-(res[n][0]-res[n][1]):+.4f}')
