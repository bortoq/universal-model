#!/usr/bin/env python3
"""Кластеризованный датасет: 64 базовых паттерна * 16 копий с шумом 1-5 бит"""
import struct, random, os
V_BYTES=32
N_BASE=64
PER_BASE=16
FLIPS=5
OUT="/tmp/universal-model/src/assoc/data/synth_clustered.bin"
random.seed(0)
bases = [bytes(random.getrandbits(8) for _ in range(V_BYTES)) for _ in range(N_BASE)]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,"wb") as f:
    n = N_BASE*PER_BASE
    f.write(struct.pack("<II", n, V_BYTES))
    for base in bases:
        for _ in range(PER_BASE):
            f.write(base)
            # noisy = base with flips
            vn = bytearray(base)
            for _ in range(random.randint(1, FLIPS)):
                bi = random.randrange(V_BYTES)
                vn[bi] ^= 1 << random.randrange(8)
            f.write(bytes(vn))
print(f"saved clustered {N_BASE}x{PER_BASE}={n} to {OUT}")
# quick check
with open(OUT,"rb") as f:
    n,vb=struct.unpack("<II",f.read(8))
    for i in range(3):
        v=f.read(vb); vn=f.read(vb)
        ham=bin(int.from_bytes(bytes(a^b for a,b in zip(v,vn)),'big')).count('1')
        print(f"ham {i}: {ham}")
