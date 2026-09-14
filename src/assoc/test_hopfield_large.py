#!/usr/bin/env python3
import os, struct, random, statistics, time
import numpy as np

V_BYTES=32; V_BITS=256
K_BITS=32; K_BYTES=4
DATA_TRAIN="/tmp/fair_A256_train.bin"
DATA_HOLD="/tmp/fair_A256_hold.bin"
CB_PQ="/tmp/hour_cb.pkl"  # PQ 256*32 from earlier? if not, create small
REPORT="/tmp/hopfield_large_report.txt"

def hamming(a,b):
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')
def bits_to_np(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))
def bits_to_pm1(b):
    arr=bits_to_np(b).astype(np.float32)
    return np.where(arr==0,-1,1)

def load_data(path, nmax=80000):
    Vs=[]
    with open(path,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        n=min(n,nmax)
        for _ in range(n):
            v=f.read(vb); vn=f.read(vb); Vs.append(v)
    return Vs

# load large
Vs_train=load_data(DATA_TRAIN, 80000)
Vs_hold=load_data(DATA_HOLD, 20000)
print(f"train {len(Vs_train)} hold {len(Vs_hold)}")

# prepare PQ for compressed test
import pickle
if os.path.exists("/tmp/cb_test.pkl"):
    cents=pickle.load(open("/tmp/cb_test.pkl","rb"))["centroids"]
else:
    cents=None

def pq_encode(v):
    # simple PQ m=8 C=64 from earlier hour_cb.pkl if exists, else nearest cents
    if cents is None:
        return v[:4] # truncate
    # use centroids 64
    best=min(range(len(cents)), key=lambda i: hamming(v, cents[i]))
    # K as index packed 4B
    return struct.pack("<I", best)[:4]

# Hopfield modern
def hopfield_retrieve(query_pm1, Mem_pm1):
    # Mem: MxD, query: D
    scores = Mem_pm1 @ query_pm1  # M
    # softmax separation
    w = np.exp(scores*0.05)  # temp 0.05 to be less peaky
    w/=w.sum()
    rec = w @ Mem_pm1  # D
    bits = (rec>0).astype(np.uint8)
    return np.packbits(bits).tobytes()

# store 500 memories
M=500
Mem_V = np.array([bits_to_pm1(v) for v in Vs_train[:M]], dtype=np.float32) # Mx256
Mem_K = np.array([bits_to_pm1(pq_encode(v).ljust(32,b'\x00')[:32]) if V_BYTES==32 else bits_to_pm1(v) for v in Vs_train[:M]], dtype=np.float32) # Mx256 but K is 4B -> 32 bits, pad to 256 for same

# Actually Mem_K should be 32 bits: need 32-dim Hopfield
Mem_K32 = np.array([bits_to_np(pq_encode(v))[:32].astype(np.float32)*2-1 for v in Vs_train[:M]])
print(f"Mem shapes {Mem_V.shape} {Mem_K32.shape}")

def test_original():
    hams=[]
    for v in Vs_hold[:2000]:
        # noisy query flip 3 bits
        vn=bytearray(v)
        for _ in range(3):
            bi=random.randrange(V_BYTES); vn[bi]^=1<<random.randrange(8)
        vn=bytes(vn)
        q=bits_to_pm1(vn)
        rec=hopfield_retrieve(q, Mem_V)
        hams.append(hamming(v, rec))
    return statistics.mean(hams), max(hams), min(hams)

def test_compressed():
    # encode hold to K, then Hopfield on K space D=32
    hams_K=[]
    hams_VviaK=[]
    for v in Vs_hold[:2000]:
        vn=bytearray(v)
        for _ in range(3):
            bi=random.randrange(V_BYTES); vn[bi]^=1<<random.randrange(8)
        vn=bytes(vn)
        k=pq_encode(v)
        kn=pq_encode(vn)
        q=np.where(bits_to_np(kn)[:32]==0,-1,1).astype(np.float32)
        # Hopfield K
        scores = Mem_K32 @ q  # M
        w=np.exp(scores*0.2); w/=w.sum()
        rec_K_pm1 = w @ Mem_K32 # 32
        rec_K=np.packbits((rec_K_pm1>0).astype(np.uint8)).tobytes()
        hams_K.append(hamming(k, rec_K))
        # decode rec_K back to V via nearest centroid
        # rec_K is 4B index hash, need to map to centroid
        # simplify: rec_K is binary 32 bits, we map via hamming to nearest centroid's K
        # find nearest original K in Mem
        # actually we have K as index 4B, rec_K is 4B binary, find nearest cent
        # for simplicity, decode via centroid lookup: find cent with K closest to rec_K
        best=None; bestd=1e9
        for c in cents or []:
            kc=pq_encode(c)
            d=hamming(rec_K, kc)
            if d<bestd:
                bestd=d; best=c
        if best is not None:
            hams_VviaK.append(hamming(v, best))
    return statistics.mean(hams_K), statistics.mean(hams_VviaK) if hams_VviaK else 0

if __name__=="__main__":
    t0=time.time()
    log=open(REPORT,"w")
    def tee(s):
        print(s); log.write(s+"\n"); log.flush()
    tee(f"test hopfield large M={M} N_hold 2000")
    ham_orig, mx, mn = test_original()
    tee(f"original V 256b Hopfield: ham mean {ham_orig:.2f} min {mn} max {mx}")
    hamK, hamVviaK = test_compressed()
    tee(f"compressed K 32b Hopfield: ham K mean {hamK:.2f} ham VviaK {hamVviaK:.2f}")
    tee(f"total {time.time()-t0:.1f}s")
    log.close()
