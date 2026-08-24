#!/usr/bin/env python3
"""pure python, без numpy - для проверки интерфейса"""
import os, random, struct

def gen(n=1000, v_bytes=32, flips=3, out="src/assoc/data/synth.bin", seed=0):
    random.seed(seed)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<II", n, v_bytes))
        for _ in range(n):
            v = bytes(random.getrandbits(8) for _ in range(v_bytes))
            f.write(v)
            # noisy
            vn = bytearray(v)
            for _ in range(random.randint(1, flips)):
                bi = random.randrange(v_bytes)
                bit = 1 << random.randrange(8)
                vn[bi] ^= bit
            f.write(bytes(vn))
    print(f"saved {n} x {v_bytes*8}bit to {out}")
    # check first 5 hamming
    with open(out, "rb") as f:
        n_r, vb = struct.unpack("<II", f.read(8))
        for i in range(min(5, n_r)):
            v = f.read(vb); vn = f.read(vb)
            ham = bin(int.from_bytes(bytes(a^b for a,b in zip(v,vn)), 'big')).count('1')
            print(f"ham {i}: {ham}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--v-bytes", type=int, default=32)
    ap.add_argument("--out", type=str, default="/tmp/universal-model/src/assoc/data/synth.bin")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gen(args.n, args.v_bytes, 3, args.out, args.seed)
