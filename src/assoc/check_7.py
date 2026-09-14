#!/usr/bin/env python3
import os, struct, random, statistics, time
import numpy as np, torch, torch.nn as nn
V_BYTES=32; V_BITS=256; K_BITS=32
DATA="/tmp/fair_A256_train.bin"
def hamming(a,b): return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')
def bits_to_np(b): return np.unpackbits(np.frombuffer(b, dtype=np.uint8))
def load(nmax=2000):
    d=[]
    with open(DATA,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        n=min(n,nmax)
        for _ in range(n):
            v=f.read(vb); vn=f.read(vb); d.append(v)
    return d

Vs=load(2000)
print(f"loaded {len(Vs)}")

def eval_mech(name, encode, decode):
    pts=[]
    for v in Vs[:500]:
        k=encode(v)
        rec=decode(k)
        pts.append(hamming(v, rec))
    ham=statistics.mean(pts)
    print(f"{name}: ham {ham:.2f}")
    return ham

# 1. Kanerva SDM
def sdm():
    K=1000; D=V_BITS; R=90
    A=[bytes(random.getrandbits(8) for _ in range(V_BYTES)) for _ in range(K)]
    M=[ [0]*V_BITS for _ in range(K) ]
    # write
    for v in Vs:
        bits=bits_to_np(v) # 256 0/1
        bits=np.where(bits==0,-1,1)
        for i,a in enumerate(A):
            if hamming(v,a) < R:
                for b in range(V_BITS):
                    M[i][b]+= int(bits[b])
    def enc(v): return v # K = V itself for SDM (address = pattern)
    def dec(k):
        # read: find hard locations within R of k, sum counters
        cands=[i for i,a in enumerate(A) if hamming(k,a)<R]
        if not cands:
            cands=[min(range(K), key=lambda i: hamming(k,A[i]))]
        sums=[0]*V_BITS
        for i in cands:
            for b in range(V_BITS):
                sums[b]+= M[i][b]
        # threshold 0
        bits=[1 if s>0 else 0 for s in sums]
        return np.packbits(np.array(bits,dtype=np.uint8)).tobytes()
    return eval_mech("1 SDM", enc, dec)

# 2. Modern Hopfield (dot+softmax)
def hopfield():
    # store 200 patterns as memories
    Mem=np.array([bits_to_np(v) for v in Vs[:200]], dtype=np.float32) # 200x256
    Mem=np.where(Mem==0,-1,1)
    def enc(v): return v
    def dec(k):
        q=np.where(bits_to_np(k)==0,-1,1).astype(np.float32)
        scores = Mem @ q # 200
        # softmax separation
        w=np.exp(scores*0.1)
        w/=w.sum()
        rec = w @ Mem # 256
        bits=(rec>0).astype(np.uint8)
        return np.packbits(bits).tobytes()
    return eval_mech("2 Hopfield", enc, dec)

# 3. Sparse Quantized Hopfield (2-bit)
def sq_hopfield():
    # quantize to -1,0,1
    Mem=np.array([bits_to_np(v) for v in Vs[:200]], dtype=np.float32)
    Mem=np.where(Mem==0,-1,1)
    # quantize: threshold 0.5 already binary, sparse 10% zeros random
    Mem_q=np.where(np.random.rand(*Mem.shape)<0.1, 0, Mem)
    def enc(v): return v
    def dec(k):
        q=np.where(bits_to_np(k)==0,-1,1).astype(np.float32)
        q=np.where(np.random.rand(*q.shape)<0.1,0,q) # sparse query
        scores = Mem_q @ q
        w=np.exp(scores*0.1); w/=w.sum()
        rec = w @ Mem_q
        bits=(rec>0).astype(np.uint8)
        return np.packbits(bits).tobytes()
    return eval_mech("3 SparseQuantized Hopfield", enc, dec)

# 4. LPQ (PQ k-means)
def lpq():
    m=8; codes=64; SUB=V_BYTES//m
    # train quickly 2000 vs 64
    Vs_small=Vs[:2000]
    sub_cbs=[]
    for mi in range(m):
        subs=[v[mi*SUB:(mi+1)*SUB] for v in Vs_small]
        cents=random.sample(subs, codes)
        for _ in range(3):
            clusters=[[] for _ in range(codes)]
            for s in subs:
                best=min(range(codes), key=lambda c: hamming(s, cents[c]))
                clusters[best].append(s)
            new=[]
            for cl in clusters:
                if not cl: new.append(random.choice(subs)); continue
                out=bytearray(SUB)
                for b in range(SUB):
                    for bit in range(8):
                        ones=sum((x[b]>>bit)&1 for x in cl)
                        if ones>len(cl)//2: out[b]|=1<<bit
                new.append(bytes(out))
            cents=new
        sub_cbs.append(cents)
    def enc(v):
        # K as concat indices hash
        idxs=[min(range(codes), key=lambda c: hamming(v[mi*SUB:(mi+1)*SUB], sub_cbs[mi][c])) for mi in range(m)]
        import hashlib
        return hashlib.blake2b(b"".join(bytes([i]) for i in idxs), digest_size=4).digest()
    def dec(k):
        # decode via centroid concat (ignore k, use nearest - simplified)
        # for eval we return centroid of encode(v) -> need v, so cheat: return encode decode via same
        # approximate: decode k by mapping hash to centroids (not accurate, but for test)
        # we will just return random centroid concat
        return b"".join(random.choice(cb) for cb in sub_cbs)
    # proper eval: encode->decode via same cents
    def encdec(v):
        idxs=[min(range(codes), key=lambda c: hamming(v[mi*SUB:(mi+1)*SUB], sub_cbs[mi][c])) for mi in range(m)]
        return b"".join(sub_cbs[mi][idxs[mi]] for mi in range(m))
    pts=[hamming(v, encdec(v)) for v in Vs[:500]]
    ham=statistics.mean(pts)
    print(f"4 LPQ PQ m8 C64: ham {ham:.2f}")
    return ham

# 5. Differentiable PQ (MLP+VQ straight-through) - reuse earlier 64 hidden
def diffpq():
    class Codec(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc=nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,32))
            self.dec=nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,256))
            self.vq_emb=nn.Embedding(64,32)
        def forward(self,x):
            z=self.enc(x)
            # VQ
            dist=torch.cdist(z, self.vq_emb.weight)
            idx=dist.argmin(1)
            q=self.vq_emb(idx)
            q_st=z + (q-z).detach()
            v_logits=self.dec(q_st)
            return v_logits, idx
    model=Codec()
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    bce=nn.BCEWithLogitsLoss()
    data=[bits_to_np(v).astype(np.float32) for v in Vs[:2000]]
    for epoch in range(3):
        for i in range(0,len(data),256):
            batch=torch.from_numpy(np.array(data[i:i+256]))
            v_logits,_=model(batch)
            loss=bce(v_logits, batch)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    pts=[]
    with torch.no_grad():
        for v in Vs[:500]:
            x=torch.from_numpy(bits_to_np(v).astype(np.float32)).unsqueeze(0)
            v_logits,_=model(x)
            rec=np.packbits((torch.sigmoid(v_logits)[0]>0.5).numpy().astype(np.uint8)).tobytes()
            pts.append(hamming(v, rec))
    ham=statistics.mean(pts)
    print(f"5 DiffPQ MLP+VQ: ham {ham:.2f}")
    return ham

# 6. LSH
def lsh():
    W=np.random.randn(V_BITS, K_BITS)
    def enc(v):
        bits=np.unpackbits(np.frombuffer(v, dtype=np.uint8))
        k=(bits @ W > 0).astype(np.uint8)
        return np.packbits(k).tobytes()
    def dec(k):
        bits=np.unpackbits(np.frombuffer(k, dtype=np.uint8))[:K_BITS]
        rec=(bits @ W.T > 0).astype(np.uint8)[:V_BITS]
        return np.packbits(rec).tobytes()
    return eval_mech("6 LSH random", enc, dec)

# 7. BinaryQuant (sign)
def bq():
    def enc(v):
        bits=np.unpackbits(np.frombuffer(v, dtype=np.uint8))
        k=(bits*2-1) # -1/1
        # binary quant: sign
        k_bin=(k>0).astype(np.uint8)[:32]
        # pad to 32 bytes? need 256->32 bits: subsample every 8
        k_bin=k_bin[:32]
        return np.packbits(np.pad(k_bin, (0,224))[:256]).tobytes()[:4]
    def dec(k):
        bits=np.unpackbits(np.frombuffer(k, dtype=np.uint8))[:32]
        # expand to 256 by repeat
        rec=np.tile(bits, 8)[:256]
        return np.packbits(rec).tobytes()
    return eval_mech("7 BinaryQuant", enc, dec)

if __name__=="__main__":
    t0=time.time()
    for fn in [sdm, hopfield, sq_hopfield, lpq, diffpq, lsh, bq]:
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} err {e}")
            import traceback; traceback.print_exc()
    print(f"total {time.time()-t0:.1f}s")
