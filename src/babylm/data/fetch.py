"""Fetch the official BabyLM 2025 text + evaluation data from OSF (node ryjfm) and verify every sha256.
Usage: python fetch.py DEST   (DEST gets text_data/ and evaluation_data/)
"""
import sys,os,hashlib,urllib.request,zipfile,socket,time,concurrent.futures as cf
socket.setdefaulttimeout(120)   # an OSF read that stalls raises instead of hanging the VM (A3 screen, 10 Oct 06:13Z)
DEST=sys.argv[1] if len(sys.argv)>1 else 'osf'
rows=[];stack=[]
for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'osf_listing.txt')):
    ind=(len(line)-len(line.lstrip(' ')))//2; s=line.strip()
    stack=stack[:ind]
    if s.startswith('[D] '): stack.append(s[4:]); continue
    name,size,sha,url=s.split('\t'); rows.append(('/'.join(stack+[name]),int(size),sha[7:],url))
def get(r):
    p,size,sha,url=r
    top='text_data' if p.split('/')[0] in('README.md','dev.zip','train_10M.zip','test.zip','train_100M.zip') else 'evaluation_data'
    out=os.path.join(DEST,top,p); os.makedirs(os.path.dirname(out),exist_ok=True)
    for i in range(6):
        try:
            if not (os.path.exists(out) and hashlib.sha256(open(out,'rb').read()).hexdigest()==sha):
                urllib.request.urlretrieve(url,out)
            if hashlib.sha256(open(out,'rb').read()).hexdigest()==sha: return (out, True)
            os.remove(out); err='sha256 mismatch'; time.sleep(2**i)   # truncated download: fetch again
        except Exception as e: err=e; time.sleep(2**i)
    return (out, f'ERR {err}')
with cf.ThreadPoolExecutor(8) as ex:
    res=list(ex.map(get,rows))
bad=[r for r in res if r[1] is not True]
print(len(res),'files', 'bad:',bad)
if bad: sys.exit(1)
# unzip text splits and the (publicly password-protected, password from the eval README) fast EWoK set
td=os.path.join(DEST,'text_data')
for z in ('train_10M','train_100M','dev','test'):
    zipfile.ZipFile(os.path.join(td,z+'.zip')).extractall(os.path.join(td,z))
fe=os.path.join(DEST,'evaluation_data','fast_eval')
zf=zipfile.ZipFile(os.path.join(fe,'ewok_fast.zip'))
for n in zf.namelist():
    if n.endswith('.jsonl'):
        os.makedirs(os.path.join(fe,'ewok_fast'),exist_ok=True)
        open(os.path.join(fe,'ewok_fast',os.path.basename(n)),'wb').write(zf.read(n,pwd=b'BabyLM2025'))
print('ok')
