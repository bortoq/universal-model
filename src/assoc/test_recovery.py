#!/usr/bin/env python3
"""Тест окрестности восстановления AoV: V -> K -> V' ошибка"""
import struct, pickle, statistics

V_BYTES=32

def hamming(a,b): return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)),'big')).count('1')

def load_centroids(pkl):
    with open(pkl,"rb") as f:
        d=pickle.load(f)
        return d["centroids"]

def test(data_path, pkl_path, topk=10):
    cents=load_centroids(pkl_path)
    def encode(v):
        return min(range(len(cents)), key=lambda c: hamming(v, cents[c]))
    # collect errors
    errors=[]
    errors_topk=[]
    same_code=0
    exact=0
    within5=0
    within10=0
    n=0
    with open(data_path,"rb") as f:
        n_total, vb = struct.unpack("<II", f.read(8))
        n=n_total
        for _ in range(n_total):
            v=f.read(vb); vn=f.read(vb)
            idx=encode(v)
            # top1 is centroid
            rec=cents[idx]
            err=hamming(v, rec)
            errors.append(err)
            if err==0: exact+=1
            if err<=5: within5+=1
            if err<=10: within10+=1
            # topk: minimal ham among topk centroids
            dists=sorted([(hamming(rec, c), c) for c in cents])
            # find best among topk
            best=min(hamming(v, c) for _, c in dists[:topk])
            errors_topk.append(best)
            # noisy same-code check
            idx_n=encode(vn)
            if idx==idx_n: same_code+=1
    errors_sorted=sorted(errors)
    errors_topk_sorted=sorted(errors_topk)
    def pct(a,p): return a[int(len(a)*p/100)]
    print(f"=== {data_path} vs {pkl_path} (codebook {len(cents)}) ===")
    print(f"n={n} V=256b K~{len(cents).bit_length()}b (codebook {len(cents)})")
    print(f"top1 centroid ham: mean={statistics.mean(errors):.2f} median={statistics.median(errors)} min={min(errors)} max={max(errors)} p90={pct(errors_sorted,90)} p95={pct(errors_sorted,95)}")
    print(f"top{topk} best ham: mean={statistics.mean(errors_topk):.2f} median={statistics.median(errors_topk)} p90={pct(errors_topk_sorted,90)}")
    print(f"exact (ham==0): {exact}/{n}={exact/n:.3f}")
    print(f"within5 (ham<=5): {within5}/{n}={within5/n:.3f}")
    print(f"within10 (ham<=10): {within10}/{n}={within10/n:.3f}")
    print(f"same-code noisy (vn same K): {same_code}/{n}={same_code/n:.3f}")
    print(f"окрестность: при top1 ошибка {statistics.mean(errors):.1f} бит из 256 ({statistics.mean(errors)/256*100:.1f}%), при top{topk} {statistics.mean(errors_topk):.1f} бит")
    print()
    return errors

if __name__=="__main__":
    test("/tmp/universal-model/src/assoc/data/synth.bin", "/tmp/universal-model/src/assoc/transformer/codebook.pkl")
    test("/tmp/universal-model/src/assoc/data/synth_clustered.bin", "/tmp/universal-model/src/assoc/transformer/codebook_clustered.pkl")
    # also test AssocMemory online decoding
    print("--- AssocMemory decode (centroid neighborhood) ---")
    from transformer.assoc_memory import AssocMemory, hamming as ham2
    import struct
    mem=AssocMemory.load("/tmp/universal-model/src/assoc/transformer/codebook_clustered.pkl")
    # For clustered, test decode neighborhood size
    with open("/tmp/universal-model/src/assoc/data/synth_clustered.bin","rb") as f:
        n,vb=struct.unpack("<II", f.read(8))
        errs=[]
        for _ in range(n):
            v=f.read(vb); vn=f.read(vb)
            k=mem.encode(v)
            neigh=mem.decode(k, topk=1)
            err=ham2(v, neigh[0])
            errs.append(err)
        print(f"AssocMemory clustered top1 ham mean={sum(errs)/len(errs):.2f} (same as above)")
