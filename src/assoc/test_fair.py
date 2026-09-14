#!/usr/bin/env python3
import os, struct, random, statistics, time
import numpy as np, torch, torch.nn as nn

N_TRAIN=80000
N_HOLD=20000
EPOCHS=20
BATCH=256
REPORT="/tmp/fair_report.txt"

def bits_to_tensor(b):
    return torch.from_numpy(np.unpackbits(np.frombuffer(b, dtype=np.uint8))).float()
def hamming(a,b):
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')

def gen(path, v_bytes, n, n_base=800):
    if os.path.exists(path):
        with open(path,"rb") as f:
            nn,_=struct.unpack("<II", f.read(8))
            if nn==n:
                print(f"exists {path}"); return
        os.remove(path)
    random.seed(1)
    bases=[bytes(random.getrandbits(8) for _ in range(v_bytes)) for _ in range(n_base)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    per=n//n_base
    with open(path,"wb") as f:
        f.write(struct.pack("<II", n, v_bytes))
        cnt=0
        while cnt<n:
            for base in bases:
                if cnt>=n: break
                f.write(base)
                vn=bytearray(base)
                for _ in range(random.randint(1,5)):
                    bi=random.randrange(v_bytes); vn[bi]^=1<<random.randrange(8)
                f.write(bytes(vn)); cnt+=1
    print(f"saved {path} n={n} v={v_bytes}")

class Codec(nn.Module):
    def __init__(self, vin, kbits, hidden):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(vin,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU(), nn.Linear(hidden,kbits))
        self.dec=nn.Sequential(nn.Linear(kbits,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU(), nn.Linear(hidden,vin))
        self.vin=vin; self.kbits=kbits
    def encode(self, x):
        return (torch.sigmoid(self.enc(x))>0.5).float(), self.enc(x)
    def forward(self, x):
        k_logits=self.enc(x)
        k=torch.sigmoid(k_logits)
        v_logits=self.dec(k)
        return k, v_logits, k_logits

def train_eval(v_bytes, k_bits, hidden, tag):
    v_bits=v_bytes*8
    train=f"/tmp/fair_{tag}_train.bin"
    hold=f"/tmp/fair_{tag}_hold.bin"
    gen(train, v_bytes, N_TRAIN)
    gen(hold, v_bytes, N_HOLD)
    # load
    def load(p, nmax=None):
        d=[]
        with open(p,"rb") as f:
            n,vb=struct.unpack("<II", f.read(8))
            n=min(n,nmax) if nmax else n
            for _ in range(n):
                v=f.read(vb); vn=f.read(vb); d.append((v,vn))
        return d
    data=load(train)
    model=Codec(v_bits, k_bits, hidden)
    opt=torch.optim.Adam(model.parameters(), lr=2e-3)
    bce=nn.BCEWithLogitsLoss()
    t0=time.time()
    for epoch in range(EPOCHS):
        random.shuffle(data)
        tot=0
        for i in range(0,len(data),BATCH):
            batch=data[i:i+BATCH]
            xv=torch.stack([bits_to_tensor(v) for v,_ in batch])
            xvn=torch.stack([bits_to_tensor(vn) for _,vn in batch])
            k, v_logits,_ = model(xv)
            kn,_,_ = model(xvn)
            loss=bce(v_logits, xv) + 0.7*((k-kn)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"{tag} epoch {epoch} loss {tot/len(data)*BATCH:.4f}")
    # eval 10k
    hold_data=load(hold, 10000)
    pts=[]; balls=[]; same=0
    model.eval()
    with torch.no_grad():
        for v,vn in hold_data:
            xv=bits_to_tensor(v).unsqueeze(0)
            xvn=bits_to_tensor(vn).unsqueeze(0)
            k,_=model.encode(xv)
            kn,_=model.encode(xvn)
            if torch.equal(k,kn): same+=1
            v_logits=model.dec(k)
            rec=np.packbits((torch.sigmoid(v_logits)[0]>0.5).cpu().numpy().astype(np.uint8)).tobytes()
            pts.append(hamming(v, rec))
            # ball 16 flips 2 bits
            best=pts[-1]
            for _ in range(16):
                kf=k.clone()
                for b in random.sample(range(k_bits),2):
                    kf[0,b]=1-kf[0,b]
                v2=model.dec(kf)
                rec2=np.packbits((torch.sigmoid(v2)[0]>0.5).cpu().numpy().astype(np.uint8)).tobytes()
                best=min(best, hamming(v, rec2))
            balls.append(best)
    res=(statistics.mean(pts), statistics.mean(balls), same/len(hold_data), time.time()-t0)
    print(f"{tag} DONE pt {res[0]:.2f} ball {res[1]:.2f} same {res[2]:.3f} time {res[3]:.1f}s")
    return res

if __name__=="__main__":
    log=open(REPORT,"w")
    def tee(s):
        print(s); log.write(s+"\n"); log.flush()
    tee(f"FAIR start {time.ctime()} N={N_TRAIN} epochs={EPOCHS} batch={BATCH}")
    tee("Config A: V256 K32 hidden512")
    a=train_eval(32,32,512,"A256")
    tee(f"A256 pt {a[0]:.2f} ball {a[1]:.2f} same {a[2]:.3f} time {a[3]:.1f}s")
    tee("Config B: V1024 K128 hidden2048 (scaled 4x)")
    b=train_eval(128,128,2048,"B1024")
    tee(f"B1024 pt {b[0]:.2f} ball {b[1]:.2f} same {b[2]:.3f} time {b[3]:.1f}s")
    tee(f"TABLE | V->K | pt ham | ball ham | same | time")
    tee(f"A256 | 256->32 | {a[0]:.2f} | {a[1]:.2f} | {a[2]:.3f} | {a[3]:.1f}s")
    tee(f"B1024| 1024->128| {b[0]:.2f} | {b[1]:.2f} | {b[2]:.3f} | {b[3]:.1f}s")
    log.close()
    print("report",REPORT)
