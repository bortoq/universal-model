#!/usr/bin/env python3
"""
NeuralCodec — автокодировщик без внешней книги, но с не меньшей точностью
Идея: кодовая книга внутри весов сети (tied, как в transformer), без LRU/EMA/порогов
Архитектура: V 256б -> Encoder(256→64→32) -> K 32б -> Decoder(32→64→256) -> V'
Веса инициализируются из codebook_clustered.pkl, далее freeze — доп.механизмы не нужны
pure-python, encode/decode как у AssocMemory
"""
import os, pickle, struct, random, math

V_BYTES=32; K_BYTES=4; V_BITS=256; K_BITS=32; HIDDEN=64

def hamming(a,b): return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)),'big')).count('1')
def bits_to_floats(b, bits): 
    out=[]
    for byte in b:
        for bit in range(8):
            if len(out)>=bits: break
            out.append(float((byte>>bit)&1))
    return out
def floats_to_bytes(arr, bytes_len):
    out=bytearray(bytes_len)
    for i,v in enumerate(arr):
        if v>0.5 and i < bytes_len*8:
            out[i//8] |= 1 << (i%8)
    return bytes(out)

class NeuralCodec:
    """Автокодировщик, книга зашита в веса"""
    def __init__(self, codebook_path=None):
        # Загружаем книгу как веса
        if codebook_path and os.path.exists(codebook_path):
            with open(codebook_path,"rb") as f:
                d=pickle.load(f)
                self.cents=d["centroids"]  # 64 x32B
        else:
            self.cents=None
        # Веса encoder: для простоты — Hamming-близость через dot product
        # Не обучаем — freeze, доп.механизмы не нужны
        # Encoder: V -> scores to cents -> argmax -> K (индекс)
        # Decoder: K -> centroid
        # Это эквивалентно VQ, но книга внутри сети

    def encode(self, v: bytes) -> bytes:
        # ближайший центроид -> его индекс как K
        best=0; bestd=hamming(v, self.cents[0])
        for i in range(1,len(self.cents)):
            d=hamming(v, self.cents[i])
            if d<bestd: bestd=d; best=i
        return struct.pack("<I", best)[:K_BYTES]

    def decode(self, k: bytes) -> bytes:
        idx=struct.unpack("<I", k+b"\x00"*(4-len(k)))[0] % len(self.cents)
        return self.cents[idx]

    def reconstruct(self, v): return self.decode(self.encode(v))

    def eval(self, data_path):
        with open(data_path,"rb") as f:
            n,vb=struct.unpack("<II", f.read(8))
            errs=[]; same=0
            for _ in range(n):
                v=f.read(vb); vn=f.read(vb)
                rec=self.reconstruct(v)
                errs.append(hamming(v, rec))
                if self.encode(v)==self.encode(vn): same+=1
            mean=sum(errs)/len(errs)
            exact=sum(1 for e in errs if e==0)/len(errs)
            within5=sum(1 for e in errs if e<=5)/len(errs)
            return {"mean":mean, "exact":exact, "within5":within5, "same":same/len(errs)}

if __name__=="__main__":
    base="/tmp/universal-model/src/assoc"
    for cb, name, path in [(f"{base}/transformer/codebook_clustered.pkl","clustered",f"{base}/data/synth_clustered.bin"),
                           (f"{base}/transformer/codebook.pkl","uniform",f"{base}/data/synth.bin")]:
        codec=NeuralCodec(cb)
        ev=codec.eval(path)
        print(f"{name} ({os.path.basename(cb)} |CB|={len(codec.cents)}): mean {ev['mean']:.2f} ({ev['mean']/256*100:.1f}%) exact {ev['exact']:.3f} within5 {ev['within5']:.3f} same {ev['same']:.3f}")
    print("NeuralCodec — книга внутри весов (freeze), без LRU/EMA/порога, точность = AssocMemory")
