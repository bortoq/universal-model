#!/usr/bin/env python3
"""
Часовой тест AoV шар->шар (PQ + ball) на больших данных. Бюджет 1 час.
N=120k train +30k holdout, PQ m=8 C=256, numpy-векторизация.
"""
import os, struct, random, pickle, statistics, time
import numpy as np

V_BYTES=32
N_TRAIN=120000
N_HOLD=30000
N_BASE=800
PER_BASE=150
FLIPS=5
CODES=256
M=8
SUB_BYTES=V_BYTES//M
R_BALL=2

DATA_TRAIN="/tmp/universal-model/src/assoc/data/hour_train.bin"
DATA_HOLD="/tmp/universal-model/src/assoc/data/hour_hold.bin"
CB_PQ="/tmp/universal-model/src/assoc/data/hour_cb.pkl"
REPORT="/tmp/hour_report.txt"

def hamming(a,b):
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')

def gen(path, n_base, per_base, n_total):
    if os.path.exists(path):
        with open(path,"rb") as f:
            n,_=struct.unpack("<II", f.read(8))
            if n==n_total:
                print(f"exists {path} n={n}")
                return
        os.remove(path)
    random.seed(0)
    bases=[bytes(random.getrandbits(8) for _ in range(V_BYTES)) for _ in range(n_base)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"wb") as f:
        f.write(struct.pack("<II", n_total, V_BYTES))
        cnt=0
        while cnt<n_total:
            for base in bases:
                if cnt>=n_total: break
                f.write(base)
                vn=bytearray(base)
                for _ in range(random.randint(1,FLIPS)):
                    bi=random.randrange(V_BYTES); vn[bi]^=1<<random.randrange(8)
                f.write(bytes(vn))
                cnt+=1
    print(f"saved {path} n={n_total}")

def load_subs(path):
    Vs=[]
    with open(path,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        for _ in range(n):
            v=f.read(vb); vn=f.read(vb)
            Vs.append(v)
    return Vs

def train_pq():
    if os.path.exists(CB_PQ):
        print(f"exists {CB_PQ}")
        return pickle.load(open(CB_PQ,"rb"))["sub_cbs"]
    print("load train...")
    Vs=load_subs(DATA_TRAIN)
    print(f"Vs {len(Vs)}")
    sub_cbs=[]
    for m in range(M):
        print(f"--- m {m} ---")
        subs=np.array([[b for b in v[m*SUB_BYTES:(m+1)*SUB_BYTES]] for v in Vs], dtype=np.uint8) # N x 4
        # init centroids random
        idx=np.random.choice(len(Vs), CODES, replace=False)
        cents=np.array([list(Vs[i][m*SUB_BYTES:(m+1)*SUB_BYTES]) for i in idx], dtype=np.uint8) # C x4
        for it in range(10):
            # assign batch 2000
            B=4000
            assign=np.zeros(len(Vs), dtype=np.int32)
            for s in range(0, len(Vs), B):
                e=min(s+B, len(Vs))
                batch=subs[s:e] # B x4
                # ham B x C : xor then popcount
                # batch[:,None,:] ^ cents[None,:,:] -> B x C x4
                xor = np.bitwise_xor(batch[:,None,:], cents[None,:,:]) # B x C x4
                # unpack bits: use bit_count via builtin popcount
                # for 4 bytes, popcount = sum of bit_count per byte: use np.unpackbits per byte
                # faster: use table 256 -> popcount
                # precompute popcount 0..255
                pop=np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
                ham = pop[xor].sum(axis=2) # B x C
                assign[s:e]=np.argmin(ham, axis=1)
            # update
            new_cents=np.zeros_like(cents)
            empty=0
            for c in range(CODES):
                mask=(assign==c)
                if not np.any(mask):
                    new_cents[c]=subs[np.random.randint(len(Vs))]
                    empty+=1
                else:
                    cl=subs[mask] # k x4
                    # majority per bit
                    # convert to bits 32
                    bits=np.unpackbits(cl, axis=1) # k x32
                    maj=(bits.sum(axis=0) > len(cl)//2).astype(np.uint8) # 32
                    # pack back to 4 bytes
                    packed=np.packbits(maj)
                    new_cents[c]=packed
            shift=(cents != new_cents).sum()/CODES
            # avg ham
            # quick avg sample 5000
            samp=np.random.choice(len(Vs), 5000, replace=False)
            samp_subs=subs[samp]
            xor_s = np.bitwise_xor(samp_subs[:,None,:], new_cents[None,:,:])
            ham_s = pop[xor_s].sum(axis=2)
            avg=np.min(ham_s, axis=1).mean()
            print(f"it{it+1} shift {shift:.2f} avg {avg:.2f} empty {empty}")
            cents=new_cents
            if shift<0.5: break
        sub_cbs.append([bytes(c) for c in cents])
    pickle.dump({"sub_cbs":sub_cbs, "M":M, "codes":CODES}, open(CB_PQ,"wb"))
    print(f"saved {CB_PQ}")
    return sub_cbs

def encode_pq_subs(v, sub_cbs):
    idxs=[]
    for m in range(M):
        sub=v[m*SUB_BYTES:(m+1)*SUB_BYTES]
        cb=sub_cbs[m]
        best=min(range(len(cb)), key=lambda c: hamming(sub, cb[c]))
        idxs.append(best)
    return idxs

if __name__=="__main__":
    import sys
    t0=time.time()
    log=open(REPORT,"w")
    def tee(s):
        print(s); log.write(s+"\n"); log.flush()
    tee(f"start {time.ctime()}")
    # force regen to match new N
    for p in [DATA_TRAIN, DATA_HOLD]:
        if os.path.exists(p):
            with open(p,"rb") as f:
                n,_=struct.unpack("<II", f.read(8))
                if n!=(N_TRAIN if 'train' in p else N_HOLD):
                    os.remove(p)
    gen(DATA_TRAIN, N_BASE, PER_BASE, N_TRAIN)
    gen(DATA_HOLD, 200, 150, N_HOLD)
    sub_cbs=train_pq()
    tee(f"train done {time.time()-t0:.1f}s")
    # eval holdout 20k sample
    N_EVAL=20000
    with open(DATA_HOLD,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        pts=[]; balls=[]
        same_pt=0; same_ball=0
        for i in range(min(N_EVAL,n)):
            v=f.read(vb); vn=f.read(vb)
            # point ham
            rec=b"".join(sub_cbs[m][encode_pq_subs(v, sub_cbs)[m]] for m in range(M))
            pts.append(hamming(v, rec))
            # ball: topk per sub r=2 -> count candidates
            # for ball we take per sub top 3 nearest
            per_sub=[]
            for m in range(M):
                sub=v[m*SUB_BYTES:(m+1)*SUB_BYTES]
                cb=sub_cbs[m]
                dists=sorted([(hamming(sub,c), idx) for idx,c in enumerate(cb)])
                cand=[idx for d,idx in dists if d<=R_BALL][:3]
                if not cand: cand=[dists[0][1]]
                per_sub.append(cand)
            # ball decode: take 16 combos (2 per sub first 2)
            import itertools
            small=[p[:2] for p in per_sub]
            cands=[]
            for combo in itertools.islice(itertools.product(*small), 16):
                vv=b"".join(sub_cbs[m][combo[m]] for m in range(M))
                cands.append(vv)
            balls.append(min(hamming(v,c) for c in cands))
            # same-code point vs ball
            idxs=encode_pq_subs(v, sub_cbs)
            idxs_n=encode_pq_subs(vn, sub_cbs)
            if idxs==idxs_n: same_pt+=1
            # ball intersect: any per_sub overlap
            per_sub_n=[]
            for m in range(M):
                sub=vn[m*SUB_BYTES:(m+1)*SUB_BYTES]
                cb=sub_cbs[m]
                dists=sorted([(hamming(sub,c), idx) for idx,c in enumerate(cb)])
                cand=[idx for d,idx in dists if d<=R_BALL][:3]
                if not cand: cand=[dists[0][1]]
                per_sub_n.append(set(cand))
            inter=sum(1 for a,b in zip(per_sub, per_sub_n) if set(a)&b)
            if inter>=M//2: same_ball+=1
            if (i+1)%5000==0:
                tee(f"eval {i+1}/{N_EVAL} pt {statistics.mean(pts):.2f} ball {statistics.mean(balls):.2f} same_pt {same_pt/(i+1):.3f} same_ball {same_ball/(i+1):.3f}")
    tee(f"holdout {N_EVAL} point ham mean {statistics.mean(pts):.2f} ball best {statistics.mean(balls):.2f}")
    tee(f"same point {same_pt/N_EVAL:.3f} same ball {same_ball/N_EVAL:.3f}")
    tee(f"total {time.time()-t0:.1f}s")
    tee(f"report {REPORT}")
    log.close()
