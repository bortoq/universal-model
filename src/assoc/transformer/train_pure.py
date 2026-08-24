#!/usr/bin/env python3
"""
Pure-python tied-VQ training без numpy/torch (т.к. Python 3.14 без wheel).
K-means на Hamming distance: V(32 bytes) -> codebook_idx -> centroid
Encode: nearest code by Hamming, Decode: centroid
Цель: noisy V' попадает в тот же кластер что и V -> recall >> hash baseline 0.03
"""
import struct, random, os, pickle

V_BYTES = 32
CODEBOOK_SIZE = 256  # |K| = log2(256)=8 бит, но храним как 1 байт + 3 padding = 4 bytes для spec |K|=32b
K_BYTES = 4

def hamming(a: bytes, b: bytes) -> int:
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')

def load_data(path):
    Vs = []
    with open(path, "rb") as f:
        n, vb = struct.unpack("<II", f.read(8))
        assert vb == V_BYTES
        for _ in range(n):
            v = f.read(vb); vn = f.read(vb)
            Vs.append(v)  # train on original V
    return Vs

def train(Vs, codebook_size=CODEBOOK_SIZE, iters=10, seed=0):
    random.seed(seed)
    # init centroids as random Vs
    centroids = random.sample(Vs, codebook_size)
    for it in range(iters):
        clusters = [[] for _ in range(codebook_size)]
        # assign
        for v in Vs:
            best = min(range(codebook_size), key=lambda c: hamming(v, centroids[c]))
            clusters[best].append(v)
        # update: centroid = per-bit majority vote
        new_centroids = []
        for idx, cl in enumerate(clusters):
            if not cl:
                new_centroids.append(random.choice(Vs))
                continue
            # majority per bit
            # convert to bit counts per position
            cent = bytearray(V_BYTES)
            for byte_pos in range(V_BYTES):
                for bit in range(8):
                    ones = sum((b[byte_pos] >> bit) & 1 for b in cl)
                    if ones > len(cl)//2:
                        cent[byte_pos] |= (1 << bit)
            new_centroids.append(bytes(cent))
        # shift measure
        shift = sum(hamming(a,b) for a,b in zip(centroids, new_centroids)) / codebook_size
        centroids = new_centroids
        # inertia
        inertia = sum(min(hamming(v,c) for c in centroids) for v in Vs) / len(Vs)
        print(f"iter {it+1}/{iters} shift={shift:.2f} avg_ham={inertia:.2f} empty={sum(1 for c in clusters if not c)}")
    return centroids

def encode(v: bytes, centroids):
    return min(range(len(centroids)), key=lambda c: hamming(v, centroids[c]))

def decode(idx: int, centroids):
    return centroids[idx]

def eval_recall(path, centroids, topk=1):
    total = 0; hit_exact = 0; hit_noisy = 0
    with open(path, "rb") as f:
        n, vb = struct.unpack("<II", f.read(8))
        for _ in range(n):
            v = f.read(vb); vn = f.read(vb)
            total += 1
            # exact: encode(v) -> decode -> compare ham
            idx = encode(v, centroids)
            rec = decode(idx, centroids)
            if rec == v:
                hit_exact += 1
            # noisy: vn should map to same cluster as v (or ham <=3 to rec)
            idx_n = encode(vn, centroids)
            rec_n = decode(idx_n, centroids)
            # check if vn's cluster centroid is close to original v (ham <=5)
            if hamming(v, rec_n) <= 5:
                hit_noisy += 1
    print(f"exact_centroid_match: {hit_exact}/{total}={hit_exact/total:.3f}")
    print(f"noisy recall (hamming<=5): {hit_noisy}/{total}={hit_noisy/total:.3f}")
    return hit_noisy/total

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/universal-model/src/assoc/data/synth.bin")
    ap.add_argument("--codebook", type=int, default=CODEBOOK_SIZE)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="/tmp/universal-model/src/assoc/transformer/codebook.pkl")
    args = ap.parse_args()
    Vs = load_data(args.data)
    print(f"loaded {len(Vs)} Vs")
    centroids = train(Vs, args.codebook, args.iters)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"centroids": centroids, "v_bytes": V_BYTES, "k_bytes": K_BYTES}, f)
    print(f"saved {len(centroids)} centroids to {args.out}")
    eval_recall(args.data, centroids)
