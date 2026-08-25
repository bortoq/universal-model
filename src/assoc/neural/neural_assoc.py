#!/usr/bin/env python3
"""
NeuralAssoc — автокодировщик без кодовой книги
V 32Б (256б) -> K 4Б (32б) -> V' 32Б
Архитектура: 256бит -> 64 hidden (ReLU) -> 32бит (sigmoid->порог) -> 64 hidden -> 256бит
pure-python, без numpy/torch, обучение — hill-climbing по расстоянию Хэмминга
Интерфейс как у AssocMemory: encode(V)->K, decode(K)->V'
"""
import random, struct, math

V_BITS=256
V_BYTES=32
K_BITS=32
K_BYTES=4
HIDDEN=16

def bits_to_floats(b: bytes):
    # 32Б -> 256 float 0/1
    out=[]
    for byte in b:
        for bit in range(8):
            out.append(float((byte>>bit)&1))
    return out  # len 256

def floats_to_bits(arr):
    # arr 0..1 threshold 0.5 -> bytes
    out=bytearray(V_BYTES)
    for i, v in enumerate(arr):
        if v>0.5:
            out[i//8] |= 1 << (i%8)
    return bytes(out)

def k_to_floats(k: bytes):
    out=[]
    for byte in k:
        for bit in range(8):
            if len(out)>=K_BITS: break
            out.append(float((byte>>bit)&1))
    return out

def floats_to_k(arr):
    out=bytearray(K_BYTES)
    for i, v in enumerate(arr[:K_BITS]):
        if v>0.5:
            out[i//8] |= 1 << (i%8)
    return bytes(out)

def mat_vec_mul(vec, mat, bias):
    # vec: n, mat: n x m, bias: m -> out m
    m=len(bias)
    n=len(vec)
    out=[0.0]*m
    for j in range(m):
        s=bias[j]
        # mat is list of lists: mat[i][j]
        for i in range(n):
            s+=vec[i]*mat[i][j]
        out[j]=s
    return out

def relu(arr): return [x if x>0 else 0.0 for x in arr]
def sigmoid(arr): return [1/(1+math.exp(-x)) for x in arr]

class NeuralAssoc:
    def __init__(self, seed=0):
        random.seed(seed)
        # W1: 256 x 64, b1:64
        self.W1=[[random.uniform(-0.5,0.5) for _ in range(HIDDEN)] for _ in range(V_BITS)]
        self.b1=[0.0]*HIDDEN
        # W2: 64 x32, b2:32
        self.W2=[[random.uniform(-0.5,0.5) for _ in range(K_BITS)] for _ in range(HIDDEN)]
        self.b2=[0.0]*K_BITS
        # W3: 32 x64, b3:64
        self.W3=[[random.uniform(-0.5,0.5) for _ in range(HIDDEN)] for _ in range(K_BITS)]
        self.b3=[0.0]*HIDDEN
        # W4: 64 x256, b4:256
        self.W4=[[random.uniform(-0.5,0.5) for _ in range(V_BITS)] for _ in range(HIDDEN)]
        self.b4=[0.0]*V_BITS

    def encode(self, v: bytes) -> bytes:
        x=bits_to_floats(v)
        h1=relu(mat_vec_mul(x, self.W1, self.b1))
        k_logits=mat_vec_mul(h1, self.W2, self.b2)
        k_sig=sigmoid(k_logits)
        return floats_to_k(k_sig)

    def decode(self, k: bytes) -> bytes:
        x=k_to_floats(k)
        h2=relu(mat_vec_mul(x, self.W3, self.b3))
        v_logits=mat_vec_mul(h2, self.W4, self.b4)
        v_sig=sigmoid(v_logits)
        return floats_to_bits(v_sig)

    def reconstruct(self, v: bytes) -> bytes:
        return self.decode(self.encode(v))

    def hamming(self, a: bytes, b: bytes) -> int:
        return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)),'big')).count('1')

    def eval(self, data_path):
        import struct
        with open(data_path,"rb") as f:
            n,vb=struct.unpack("<II", f.read(8))
            errs=[]
            same=0
            for _ in range(n):
                v=f.read(vb); vn=f.read(vb)
                rec=self.reconstruct(v)
                err=self.hamming(v, rec)
                errs.append(err)
                if self.encode(v)==self.encode(vn):
                    same+=1
            mean=sum(errs)/len(errs)
            exact=sum(1 for e in errs if e==0)
            within5=sum(1 for e in errs if e<=5)
            return {"mean":mean, "exact":exact/len(errs), "within5":within5/len(errs), "same":same/len(errs), "errs":errs}

    def train_step(self, v: bytes, lr=0.01):
        # один шаг SGD численно (очень упрощенно): пробуем сдвинуть веса в сторону уменьшения ошибки
        # Для pure-python делаем случайную мутацию и оставляем если лучше
        rec=self.reconstruct(v)
        err=self.hamming(v, rec)
        # мутация одного случайного веса
        choice=random.choice(["W1","W2","W3","W4"])
        W=getattr(self, choice)
        i=random.randrange(len(W))
        j=random.randrange(len(W[0]))
        old=W[i][j]
        W[i][j]+=random.uniform(-0.1,0.1)
        rec2=self.reconstruct(v)
        err2=self.hamming(v, rec2)
        if err2<err:
            return True  # keep
        else:
            W[i][j]=old
            return False

def train_on_data(data_path, steps=200, seed=0):
    import struct
    net=NeuralAssoc(seed)
    with open(data_path,"rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        Vs=[]
        for _ in range(n):
            v=f.read(vb); vn=f.read(vb)
            Vs.append(v)
    for s in range(steps):
        v=random.choice(Vs)
        net.train_step(v)
        if s%100==0:
            ev=net.eval(data_path)
            print(f"step {s}: mean {ev['mean']:.1f} exact {ev['exact']:.3f} same {ev['same']:.3f}")
    return net

if __name__=="__main__":
    import os, sys
    data="/tmp/universal-model/src/assoc/data/synth_clustered.bin"
    print("train clustered 200 steps...")
    net=train_on_data(data, steps=200)
    ev=net.eval(data)
    print(f"final clustered: mean {ev['mean']:.2f} exact {ev['exact']:.3f} same {ev['same']:.3f}")
    data_u="/tmp/universal-model/src/assoc/data/synth.bin"
    ev2=net.eval(data_u)
    print(f"on uniform (same net): mean {ev2['mean']:.2f} exact {ev2['exact']:.3f}")
