#!/usr/bin/env python3
import os, struct, random, statistics, time
import numpy as np, torch, torch.nn as nn
V_BYTES=128
V_BITS=1024
K_BITS=128
K_BYTES=16
N_TRAIN=80000
N_HOLD=20000
REPORT="/tmp/hour_1024_report.txt"
DATA_TRAIN="/tmp/universal-model/src/assoc/data/hour_1024_train.bin"
DATA_HOLD="/tmp/universal-model/src/assoc/data/hour_1024_hold.bin"

def hamming(a,b):
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')
def bits_to_tensor(b):
    return torch.from_numpy(np.unpackbits(np.frombuffer(b, dtype=np.uint8))).float()
def gen(path, n):
    if os.path.exists(path):
        with open(path,"rb") as f:
            nn,_=struct.unpack("<II", f.read(8))
            if nn==n:
                print(f"exists {path}")
                return
        os.remove(path)
    random.seed(1)
    bases=[bytes(random.getrandbits(8) for _ in range(V_BYTES)) for _ in range(800)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"wb") as f:
        f.write(struct.pack("<II", n, V_BYTES))
        for _ in range(n):
            base=random.choice(bases)
            f.write(base)
            vn=bytearray(base)
            for _ in range(random.randint(1,5)):
                bi=random.randrange(V_BYTES); vn[bi]^=1<<random.randrange(8)
            f.write(bytes(vn))
    print(f"saved {path} n={n}")

class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(1024,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(), nn.Linear(512,128))
        self.dec=nn.Sequential(nn.Linear(128,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(), nn.Linear(512,1024))
    def encode(self, x):
        return (torch.sigmoid(self.enc(x))>0.5).float(), self.enc(x)
    def forward(self, x):
        k_logits=self.enc(x)
        k=torch.sigmoid(k_logits)
        v_logits=self.dec(k)
        return k, v_logits, k_logits

def train():
    # load
    Vs=[]
    with open(DATA_TRAIN,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        for _ in range(n):
            v=f.read(vb); vn=f.read(vb); Vs.append((v,vn))
    model=Codec()
    opt=torch.optim.Adam(model.parameters(), lr=2e-3)
    bce=nn.BCEWithLogitsLoss()
    for epoch in range(15):
        random.shuffle(Vs)
        tot=0
        for i in range(0,len(Vs),256):
            batch=Vs[i:i+256]
            xv=torch.stack([bits_to_tensor(v) for v,_ in batch])
            xvn=torch.stack([bits_to_tensor(vn) for _,vn in batch])
            k, v_logits,_ = model(xv)
            kn,_,_ = model(xvn)
            loss_rec=bce(v_logits, xv)
            loss_ball=((k-kn)**2).mean()
            loss=loss_rec + 0.7*loss_ball
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"epoch {epoch} loss {tot/len(Vs)*256:.4f}")
    torch.save(model.state_dict(), "/tmp/codec1024.pt")
    return model

def eval_model(model):
    model.eval()
    Vs=[]
    with open(DATA_HOLD,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        for _ in range(min(n,20000)):
            v=f.read(vb); vn=f.read(vb); Vs.append((v,vn))
    pts=[]; balls=[]; same=0
    with torch.no_grad():
        for v,vn in Vs:
            xv=bits_to_tensor(v).unsqueeze(0)
            xvn=bits_to_tensor(vn).unsqueeze(0)
            k,_=model.encode(xv)
            kn,_=model.encode(xvn)
            if torch.equal(k,kn): same+=1
            v_logits=model.dec(k)
            rec=np.packbits((torch.sigmoid(v_logits)[0]>0.5).cpu().numpy().astype(np.uint8)).tobytes()
            pts.append(hamming(v, rec))
            # ball: flip 2 bits in K
            best=pts[-1]
            for _ in range(16):
                kf=k.clone()
                for b in random.sample(range(128),2):
                    kf[0,b]=1-kf[0,b]
                v2=model.dec(kf)
                rec2=np.packbits((torch.sigmoid(v2)[0]>0.5).cpu().numpy().astype(np.uint8)).tobytes()
                best=min(best, hamming(v, rec2))
            balls.append(best)
    print(f"point ham {statistics.mean(pts):.2f} ball {statistics.mean(balls):.2f} same {same/len(Vs):.3f}")
    return pts,balls

if __name__=="__main__":
    import time as tm
    t0=tm.time()
    log=open(REPORT,"w")
    def tee(s):
        print(s); log.write(s+"\n"); log.flush()
    tee(f"start {time.ctime()}")
    gen(DATA_TRAIN, N_TRAIN)
    gen(DATA_HOLD, N_HOLD)
    m=train()
    tee(f"train done {tm.time()-t0:.1f}s")
    pts,balls=eval_model(m)
    tee(f"point ham mean {statistics.mean(pts):.2f} ball {statistics.mean(balls):.2f} total {tm.time()-t0:.1f}s")
    log.close()
