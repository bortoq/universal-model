"""
Baseline pure-python для проверки интерфейса spec.md без torch/numpy.
encode: V(32 bytes) -> K(4 bytes) через blake2b, decode: поиск по Hamming в таблице.
"""
import hashlib, struct, os

def encode(v: bytes) -> bytes:
    assert len(v) == 32, "|V| должен быть 32 байта (256 бит)"
    return hashlib.blake2b(v, digest_size=4).digest()  # |K|=4 байта =32 бит

def build_table(bin_path):
    table = {}  # K -> list[V]
    with open(bin_path, "rb") as f:
        n, vb = struct.unpack("<II", f.read(8))
        for _ in range(n):
            v = f.read(vb); vn = f.read(vb)  # vn не нужен для таблицы, но читаем
            k = encode(v)
            table.setdefault(k, []).append(v)
    return table

def hamming(a: bytes, b: bytes) -> int:
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')

def decode(key: bytes, table: dict, topk=10):
    # brute force: ищем ближайшие V по Hamming к любому V с таким K (если коллизий нет — ищем ближайший K)
    # для baseline: перебираем все V в таблице
    candidates = []
    for k, vs in table.items():
        # расстояние между ключами как Hamming тоже
        kd = hamming(key, k)
        for v in vs:
            candidates.append((kd, v))
    candidates.sort(key=lambda x: x[0])
    return [v for _, v in candidates[:topk]]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/universal-model/src/assoc/data/synth.bin")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    table = build_table(args.data)
    print(f"table keys: {len(table)}")
    # test recall
    with open(args.data, "rb") as f:
        n, vb = struct.unpack("<II", f.read(8))
        hits = 0; total = min(100, n)
        for i in range(total):
            v = f.read(vb); vn = f.read(vb)
            k = encode(v)
            rec = decode(k, table, args.topk)
            if v in rec:
                hits += 1
        print(f"recall@{args.topk} (hash baseline, должен быть 1.0): {hits}/{total} = {hits/total:.2f}")
        # noisy test: encode noisy -> должен попасть в окрестность исходного
        f.seek(8)
        hits_noisy = 0
        for i in range(total):
            v = f.read(vb); vn = f.read(vb)
            k_noisy = encode(vn)
            rec = decode(k_noisy, table, args.topk)
            # проверяем Hamming <=3
            if any(hamming(v, r) <= 3 for r in rec):
                hits_noisy += 1
        print(f"noisy recall (vn->окрестность v, ham<=3) @{args.topk}: {hits_noisy}/{total} = {hits_noisy/total:.2f} (ожидаемо низко для хеша — нужен обучаемый VQ)")
