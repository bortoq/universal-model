#!/usr/bin/env python3
import os, struct, random, statistics, time
import numpy as np, torch, torch.nn as nn
V_BYTES=32
N_TRAIN=120000
N_HOLD=30000
REPORT="/tmp/hour_nocodebook_v2_report.txt"
DATA_TRAIN="/tmp/universal-model/src/assoc/data/hour_train.bin"
DATA_HOLD="/tmp/universal-model/src/assoc/data/hour_hold.bin"

def hamming(a,b):
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')
def bits_to_tensor(b):
    return torch.from_numpy(np.unpackbits(np.frombuffer(b, dtype=np.uint8))).float()
def load_data(path, n_max=None):
    Vs=[]
    with open(path,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        n=min(n, n_max) if n_max else n
        for _ in range(n):
            v=f.read(vb); vn=f.read(vb)
            Vs.append((v,vn))
    return Vs

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

def train():
    data=load_data(DATA_TRAIN)
    model=Codec()
    opt=torch.optim.Adam(model.parameters(), lr=2e-3)
    bce=nn.BCEWithLogitsLoss()
    for epoch in range(20):
        random.shuffle(data)
        tot=0
        for i in range(0,len(data),512):
            batch=data[i:i+512]
            xv=torch.stack([bits_to_tensor(v) for v,_ in batch])
            xvn=torch.stack([bits_to_tensor(vn) for _,vn in batch])
            k, v_logits,_ = model(xv)
            kn,_,_ = model(xvn)
            loss_rec=bce(v_logits, xv)
            loss_ball=((k-kn)**2).mean()
            loss=loss_rec + 0.7*loss_ball
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"epoch {epoch} loss {tot/len(data)*512:.4f}")
    torch.save(model.state_dict(), "/tmp/codec_v2.pt")
    return model

def eval_model(model):
    model.eval()
    data=load_data(DATA_HOLD, 20000)
    pts=[]; balls=[]; same_pt=0; same_ball=0
    with torch.no_grad():
        for v,vn in data:
            xv=bits_to_tensor(v).unsqueeze(0)
            xvn=bits_to_tensor(vn).unsqueeze(0)
            k,_=model.encode(xv)
            kn,_=model.encode(xvn)
            # decode
            v_logits=model.dec(k)
            rec=np.packbits((torch.sigmoid(v_logits)[0]>0.5).cpu().numpy().astype(np.uint8)).tobytes()
            pts.append(hamming(v, rec))
            if torch.equal(k,kn): same_pt+=1
            hamk=int((k[0]!=kn[0]).sum().item())
            if hamk<=1: same_ball+=1
            # ball best 16 flips
            best=pts[-1]
            for _ in range(16):
                kf=k.clone()
                for b in random.sample(range(32),2):
                    kf[0,b]=1-kf[0,b]
                v2=model.dec(kf)
                rec2=np.packbits((torch.sigmoid(v2)[0]>0.5).cpu().numpy().astype(np.uint8)).tobytes()
                best=min(best, hamming(v, rec2))
            balls.append(best)
    print(f"point ham {statistics.mean(pts):.2f} ball {statistics.mean(balls):.2f} same_pt {same_pt/len(data):.3f} same_ball {same_ball/len(data):.3f}")
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
