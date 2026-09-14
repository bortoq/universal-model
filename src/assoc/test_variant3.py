#!/usr/bin/env python3
import os, struct, random, statistics, time
import numpy as np, torch, torch.nn as nn
V_BYTES=32
V_BITS=256
K_BITS=32
N_TRAIN=80000
N_HOLD=20000
REPORT="/tmp/variant3_report.txt"
DATA_TRAIN="/tmp/fair_A256_train.bin"
DATA_HOLD="/tmp/fair_A256_hold.bin"

def bits_to_tensor(b):
    return torch.from_numpy(np.unpackbits(np.frombuffer(b, dtype=np.uint8))).float()
def hamming(a,b):
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')

# ensure data exists (reuse fair 256)
if not os.path.exists(DATA_TRAIN):
    import subprocess
    subprocess.run(["python3","-c", "import src.assoc.test_fair"], timeout=5)

class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(256,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(), nn.Linear(512,32))
        self.dec=nn.Sequential(nn.Linear(32,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(), nn.Linear(512,256))
    def encode(self, x):
        return (torch.sigmoid(self.enc(x))>0.5).float(), self.enc(x)
    def forward(self, x):
        k_logits=self.enc(x)
        k=torch.sigmoid(k_logits)
        v_logits=self.dec(k)
        return k, v_logits, k_logits

def load_data(path, nmax=None):
    d=[]
    with open(path,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        n=min(n,nmax) if nmax else n
        for _ in range(n):
            v=f.read(vb); vn=f.read(vb); d.append((v,vn))
    return d

def train():
    data=load_data(DATA_TRAIN)
    # also load negatives: random other Vs
    all_v=[v for v,_ in data]
    model=Codec()
    opt=torch.optim.Adam(model.parameters(), lr=2e-3)
    bce=nn.BCEWithLogitsLoss()
    for epoch in range(20):
        random.shuffle(data)
        tot=0
        for i in range(0,len(data),256):
            batch=data[i:i+256]
            xv=torch.stack([bits_to_tensor(v) for v,_ in batch])
            xvn=torch.stack([bits_to_tensor(vn) for _,vn in batch])
            # negative: random other V from dataset
            neg_idx=np.random.choice(len(all_v), len(batch))
            xv_neg=torch.stack([bits_to_tensor(all_v[j]) for j in neg_idx])
            k, v_logits,_ = model(xv)
            kn,_,_ = model(xvn)
            k_neg,_,_ = model(xv_neg)
            loss_rec=bce(v_logits, xv)
            # contrast: ham(K,Kn) small, ham(K,Kneg) large, margin 8 (for 32 bits)
            ham_pos=((k-kn)**2).sum(dim=1).mean()
            ham_neg=((k-k_neg)**2).sum(dim=1).mean()
            # want ham_pos < 2, ham_neg > 8
            loss_contrast= ham_pos + torch.clamp(8 - ham_neg, min=0)
            loss=loss_rec + 0.5*loss_contrast
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"epoch {epoch} loss {tot/len(data)*256:.4f}")
    torch.save(model.state_dict(), "/tmp/codec_variant3.pt")
    return model

def eval_model(model):
    data=load_data(DATA_HOLD, 10000)
    pts=[]; balls=[]; same=0
    model.eval()
    with torch.no_grad():
        for v,vn in data:
            xv=bits_to_tensor(v).unsqueeze(0)
            xvn=bits_to_tensor(vn).unsqueeze(0)
            k,_=model.encode(xv)
            kn,_=model.encode(xvn)
            if torch.equal(k,kn): same+=1
            v_logits=model.dec(k)
            rec=np.packbits((torch.sigmoid(v_logits)[0]>0.5).cpu().numpy().astype(np.uint8)).tobytes()
            pts.append(hamming(v, rec))
            best=pts[-1]
            for _ in range(16):
                kf=k.clone()
                for b in random.sample(range(32),2):
                    kf[0,b]=1-kf[0,b]
                v2=model.dec(kf)
                rec2=np.packbits((torch.sigmoid(v2)[0]>0.5).cpu().numpy().astype(np.uint8)).tobytes()
                best=min(best, hamming(v, rec2))
            balls.append(best)
    print(f"point ham {statistics.mean(pts):.2f} ball {statistics.mean(balls):.2f} same {same/len(data):.3f}")
    return pts,balls

if __name__=="__main__":
    t0=time.time()
    log=open(REPORT,"w")
    def tee(s):
        print(s); log.write(s+"\n"); log.flush()
    tee(f"start {time.ctime()}")
    m=train()
    tee(f"train done {time.time()-t0:.1f}s")
    pts,balls=eval_model(m)
    tee(f"point ham mean {statistics.mean(pts):.2f} ball {statistics.mean(balls):.2f} total {time.time()-t0:.1f}s")
    log.close()
