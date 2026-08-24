#!/usr/bin/env python3
"""eval recall@k для assoc"""
import numpy as np

def recall_at_k(V, V_rec, k=10):
    # упрощенно: считаем что decode возвращает 1 вариант, recall@1 = доля точных восстановлений по Hamming <= threshold
    # для baseline: Hamming distance
    ham = np.unpackbits(np.bitwise_xor(V, V_rec), axis=1).sum(axis=1)
    # recall@1 если ham ==0, recall@10 если ham <=3 (окрестность)
    r1 = (ham == 0).mean()
    r10 = (ham <= 3).mean()
    return r1, r10, ham.mean()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()
    d = np.load(args.npz)
    V = d["V"]
    # dummy: V_rec = V_noisy
    V_rec = d["V_noisy"] if "V_noisy" in d else V
    r1, r10, m = recall_at_k(V, V_rec)
    print(f"recall@1={r1:.3f} recall@10(th<=3)={r10:.3f} mean_ham={m:.2f}")
