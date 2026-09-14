#!/usr/bin/env python3
"""
Без словаря: V 256б -> K 32б прямой хеш MLP, K->V' MLP.
Обучение 120k, тест 20k, ~15 мин.
"""
import os, struct, random, pickle, statistics, time
import numpy as np
import torch
import torch.nn as nn

V_BITS=256
V_BYTES=32
K_BITS=32
K_BYTES=4
N_TRAIN=120000
N_HOLD=30000
REPORT="/tmp/hour_nocodebook_report.txt"
DATA_TRAIN="/tmp/universal-model/src/assoc/data/hour_train.bin"
DATA_HOLD="/tmp/universal-model/src/assoc/data/hour_hold.bin"

def hamming(a,b):
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')

def bits_to_tensor(b):
    arr=np.unpackbits(np.frombuffer(b, dtype=np.uint8)) # 256
    return torch.from_numpy(arr).float()

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
        self.enc = nn.Sequential(
            nn.Linear(256,64), nn.ReLU(),
            nn.Linear(64,32)
        )
        self.dec = nn.Sequential(
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,256)
        )
    def encode(self, x): # x Bx256 0/1
        logits=self.enc(x)
        return (torch.sigmoid(logits)>0.5).float(), logits
    def decode(self, k): # k Bx32 0/1
        logits=self.dec(k)
        return (torch.sigmoid(logits)>0.5).float(), logits
    def forward(self, x):
        k, k_logits = self.encode(x)
        # straight-through for training: use logits
        v_logits=self.dec(k).float() if False else self.dec(torch.sigmoid(k_logits)).float()
        # actually for training use continuous
        k_cont=torch.sigmoid(k_logits)
        v_cont_logits=self.dec(k_cont)
        return k_cont, v_cont_logits, k, k_logits

def train():
    data=load_data(DATA_TRAIN)
    model=Codec()
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    bce=nn.BCEWithLogitsLoss()
    # contrastive weight
    for epoch in range(5):
        random.shuffle(data)
        total=0
        for i in range(0, len(data), 1024):
            batch=data[i:i+1024]
            xv=torch.stack([bits_to_tensor(v) for v,_ in batch])
            # target is xv itself
            k_cont, v_logits, _, _ = model(xv)
            loss_rec=bce(v_logits, xv)
            # ball loss: noisy should have close K
            # take vn
            xvn=torch.stack([bits_to_tensor(vn) for _,vn in batch])
            k_cont_n,_ ,_,_ = model(xvn)
            # ham K ~ mse
            loss_ball = ((k_cont - k_cont_n)**2).mean()
            loss=loss_rec + 0.5*loss_ball
            opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item()
        print(f"epoch {epoch} loss {total/len(data)*1024:.4f}")
    torch.save(model.state_dict(), "/tmp/codec.pt")
    return model

def eval_model(model):
    model.eval()
    data=load_data(DATA_HOLD, 20000)
    pts=[]; balls=[]
    same_pt=0; same_ball=0
    with torch.no_grad():
        for v,vn in data:
            xv=bits_to_tensor(v).unsqueeze(0)
            xvn=bits_to_tensor(vn).unsqueeze(0)
            k,_ = model.encode(xv)
            kn,_ = model.encode(xvn)
            # decode point
            v_rec,_ = model.decode(k)
            rec_bytes=np.packbits(v_rec[0].cpu().numpy().astype(np.uint8)).tobytes()
            pts.append(hamming(v, rec_bytes))
            if torch.equal(k, kn): same_pt+=1
            # ball: K flip 1 bit -> 32 variants, take best ham
            k_np=k[0].cpu().numpy().astype(np.uint8) # 32
            best=pts[-1]
            # check ball intersect: any 1-bit flip of k equals kn?
            # kn is 32 bits
            kn_np=kn[0].cpu().numpy()
            # ham K
            ham_k=int((k_np!=kn_np).sum())
            if ham_k<=1: same_ball+=1
            # ball best: try 8 random 1-bit flips
            for _ in range(16):
                kf=k.clone()
                # flip random bit
                bit=random.randrange(32)
                kf[0,bit]=1-kf[0,bit]
                v_kf,_=model.decode(kf)
                rec2=np.packbits(v_kf[0].cpu().numpy().astype(np.uint8)).tobytes()
                best=min(best, hamming(v, rec2))
            balls.append(best)
    print(f"point ham mean {statistics.mean(pts):.2f} ball best {statistics.mean(balls):.2f}")
    print(f"same point {same_pt/len(data):.3f} same ball {same_ball/len(data):.3f}")
    return pts,balls

if __name__=="__main__":
    t0=time.time()
    log=open(REPORT,"w")
    def tee(s):
        print(s); log.write(s+"\n"); log.flush()
    tee(f"start {time.ctime()}")
    model=train()
    tee(f"train done {time.time()-t0:.1f}s")
    pts,balls=eval_model(model)
    tee(f"point ham mean {statistics.mean(pts):.2f} median {statistics.median(pts)}")
    tee(f"ball best ham mean {statistics.mean(balls):.2f}")
    tee(f"total {time.time()-t0:.1f}s")
    log.close()
