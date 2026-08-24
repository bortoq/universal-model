#!/usr/bin/env python3
"""
Генератор датасета для assoc V->K.
Синтетика: random V (256 бит) + noisy copies (flip 1-5 бит) — окрестность по Hamming.
Формат: npz с V [N,32] uint8, V_noisy [N,32]
"""
import argparse
import numpy as np

def gen_random_vs(n, v_bytes=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, v_bytes), dtype=np.uint8)

def add_noise(V, flips=3, seed=1):
    rng = np.random.default_rng(seed)
    out = V.copy()
    n = V.shape[0]
    for i in range(n):
        bits_to_flip = rng.integers(1, flips+1)
        for _ in range(bits_to_flip):
            byte_idx = rng.integers(0, V.shape[1])
            bit_idx = rng.integers(0, 8)
            out[i, byte_idx] ^= (1 << bit_idx)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="число V")
    ap.add_argument("--v-bytes", type=int, default=32, help="|V| в байтах (256 бит =32)")
    ap.add_argument("--flips", type=int, default=3, help="max бит для flip в noisy")
    ap.add_argument("--out", type=str, default="src/assoc/data/synth.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    V = gen_random_vs(args.n, args.v_bytes, args.seed)
    V_noisy = add_noise(V, args.flips, args.seed+1)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, V=V, V_noisy=V_noisy, v_bytes=args.v_bytes)
    print(f"saved {args.n} x {args.v_bytes*8}bit to {args.out}")
    # sanity: hamming
    ham = np.unpackbits(np.bitwise_xor(V[:5], V_noisy[:5]), axis=1).sum(axis=1)
    print(f"example hamming V vs noisy (first 5): {ham.tolist()}")

if __name__ == "__main__":
    main()
