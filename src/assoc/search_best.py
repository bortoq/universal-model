#!/usr/bin/env python3
import os, struct, random, statistics, time, pickle
import numpy as np, torch, torch.nn as nn
V_BYTES=32
V_BITS=256
K_BITS=32
REPORT="/tmp/search_best_report.txt"
DATA_TRAIN="/tmp/fair_A256_train.bin"
DATA_HOLD="/tmp/fair_A256_hold.bin"

def hamming(a,b):
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')
def bits_to_tensor(b):
    return torch.from_numpy(np.unpackbits(np.frombuffer(b, dtype=np.uint8))).float()

def load_data(path, nmax=20000):
    d=[]
    with open(path,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        n=min(n,nmax)
        for _ in range(n):
            v=f.read(vb); vn=f.read(vb); d.append((v,vn))
    return d

# mechanism A: PQ
def eval_pq(codes, m=8):
    from collections import defaultdict
    import pickle, random
    # train quickly using existing cb or train small
    # reuse hour_cb.pkl for 256, else train
    cb_path=f"/tmp/cb_{codes}_{m}.pkl"
    if not os.path.exists(cb_path):
        # train PQ m*C
        Vs=[]
        with open(DATA_TRAIN,"rb") as f:
            n,vb=struct.unpack("<II", f.read(8))
            for _ in range(min(20000,n)):
                v=f.read(vb); vn=f.read(vb); Vs.append(v)
        SUB=len(Vs[0])//m
        sub_cbs=[]
        for mi in range(m):
            subs=[v[mi*SUB_BYTES:(mi+1)*SUB_BYTES] for v in Vs] if (SUB_BYTES:=V_BYTES//m) else []
            # k-means Hamming
            cents=random.sample(subs, codes)
            for it in range(5):
                clusters=[[] for _ in range(codes)]
                for s in subs:
                    best=min(range(codes), key=lambda c: hamming(s, cents[c]))
                    clusters[best].append(s)
                new=[]
                for cl in clusters:
                    if not cl:
                        new.append(random.choice(subs)); continue
                    out=bytearray(SUB_BYTES)
                    for b in range(SUB_BYTES):
                        for bit in range(8):
                            ones=sum((x[b]>>bit)&1 for x in cl)
                            if ones>len(cl)//2: out[b]|=1<<bit
                    new.append(bytes(out))
                cents=new
            sub_cbs.append(cents)
        pickle.dump({"sub_cbs":sub_cbs}, open(cb_path,"wb"))
    else:
        sub_cbs=pickle.load(open(cb_path,"rb"))["sub_cbs"]
    # eval
    data=load_data(DATA_HOLD, 5000)
    SUB_BYTES=V_BYTES//m
    pts=[]
    for v,vn in data:
        # encode: per sub nearest
        rec=b"".join(sub_cbs[mi][min(range(codes), key=lambda c: hamming(v[mi*SUB_BYTES:(mi+1)*SUB_BYTES], sub_cbs[mi][c]))] for mi in range(m))
        pts.append(hamming(v, rec))
    return statistics.mean(pts), f"PQ m={m} C={codes}"

# mechanism B: LSH random
def eval_lsh():
    random.seed(0); np.random.seed(0)
    W=np.random.randn(V_BITS, K_BITS)
    def encode(v):
        bits=np.unpackbits(np.frombuffer(v, dtype=np.uint8))
        k=(bits @ W > 0).astype(np.uint8)
        return np.packbits(k).tobytes()
    def decode(k):
        bits=np.unpackbits(np.frombuffer(k, dtype=np.uint8))[:K_BITS]
        # pseudo-inverse: W^T * k
        rec_bits = (bits @ W.T > 0).astype(np.uint8)[:V_BITS]
        return np.packbits(rec_bits).tobytes()
    data=load_data(DATA_HOLD, 2000)
    pts=[]
    for v,_ in data:
        k=encode(v)
        rec=decode(k)
        pts.append(hamming(v, rec))
    return statistics.mean(pts), "LSH random 256->32"

# mechanism C: neural
def eval_nn(hidden, epochs):
    class Codec(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc=nn.Sequential(nn.Linear(256,hidden), nn.ReLU(), nn.Linear(hidden,32))
            self.dec=nn.Sequential(nn.Linear(32,hidden), nn.ReLU(), nn.Linear(hidden,256))
        def encode(self,x): return (torch.sigmoid(self.enc(x))>0.5).float()
    # train quickly on 20k
    data=load_data(DATA_TRAIN, 20000)
    model=Codec()
    opt=torch.optim.Adam(model.parameters(), lr=2e-3)
    bce=nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        random.shuffle(data)
        for i in range(0,len(data),512):
            batch=data[i:i+512]
            xv=torch.stack([bits_to_tensor(v) for v,_ in batch])
            k_logits=model.enc(xv)
            k=torch.sigmoid(k_logits)
            v_logits=model.dec(k)
            loss=bce(v_logits, xv)
            opt.zero_grad(); loss.backward(); opt.step()
    # eval
    hold=load_data(DATA_HOLD, 2000)
    pts=[]
    model.eval()
    with torch.no_grad():
        for v,_ in hold:
            xv=bits_to_tensor(v).unsqueeze(0)
            k=model.encode(xv)
            v_logits=model.dec(k)
            rec=np.packbits((torch.sigmoid(v_logits)[0]>0.5).cpu().numpy().astype(np.uint8)).tobytes()
            pts.append(hamming(v, rec))
    return statistics.mean(pts), f"NN hidden={hidden} epochs={epochs}"

if __name__=="__main__":
    import time as tm
    t0=tm.time()
    log=open(REPORT,"w")
    def tee(s):
        print(s); log.write(s+"\n"); log.flush()
    tee(f"search start {time.ctime()}")
    results=[]
    for codes in [64,256,1024]:
        try:
            m,desc=eval_pq(codes,8)
            tee(f"{desc}: ham {m:.2f}")
            results.append((m,desc))
        except Exception as e:
            tee(f"PQ {codes} err {e}")
    for h,e in [(64,5),(512,5),(512,20)]:
        try:
            m,desc=eval_nn(h,e)
            tee(f"{desc}: ham {m:.2f}")
            results.append((m,desc))
        except Exception as e:
            tee(f"NN {h}/{e} err {e}")
    try:
        m,desc=eval_lsh()
        tee(f"{desc}: ham {m:.2f}")
        results.append((m,desc))
    except Exception as e:
        tee(f"LSH err {e}")
    results.sort()
    tee("=== BEST ===")
    for ham,desc in results:
        tee(f"{ham:.2f} {desc}")
    tee(f"best {results[0][1]} ham {results[0][0]:.2f} total {tm.time()-t0:.1f}s")
    log.close()
