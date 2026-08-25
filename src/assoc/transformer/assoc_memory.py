#!/usr/bin/env python3
"""
AssocMemory — механизм AoV с online дообучением.

Интерфейс как в spec.md:
  encode(V: bytes[32]) -> K: bytes[4]  (|K|<|V|, сжатие 8x)
  decode(K: bytes[4]) -> list[bytes]   (окрестность)
  update(V: bytes) — EMA-дообучение без переобучения encoder

Свойства:
- tied codebook: один на всю будущую пирамиду S0->S1->...
- EMA: centroid = (1-α)*centroid + α*V, α=0.01
- LRU + порог новизны: если ham(V, nearest) > threshold — заменяем наименее используемый код
- pure-python, без numpy/torch, работает на Python 3.14

Использование как механизма:
  mem = AssocMemory.load("codebook_clustered.pkl")
  k = mem.encode(v)
  neighbors = mem.decode(k)  # top-k
  mem.update(v_new)          # дообучился на лету
"""
import struct, pickle, random
from collections import Counter

V_BYTES = 32
K_BYTES = 4

def hamming(a: bytes, b: bytes) -> int:
    return bin(int.from_bytes(bytes(x ^ y for x, y in zip(a, b)), 'big')).count('1')

def majority_update(centroid: bytes, v: bytes, alpha: float = 0.01) -> bytes:
    # EMA по битам: с вероятностью alpha берем бит из v, иначе из centroid
    # Детерминированная версия: per-byte majority с весом
    # Упрощение: побайтово, если alpha мало — меняем 1 бит с вероятностью alpha*V_BYTES*8
    # Для pure-python делаем стохастически:
    out = bytearray(centroid)
    if random.random() < alpha * 8:  # в среднем 8% вызовов меняем 1 случайный бит к v
        # выбираем случайный бит где centroid != v
        diffs = []
        for i in range(V_BYTES):
            d = centroid[i] ^ v[i]
            for bit in range(8):
                if (d >> bit) & 1:
                    diffs.append((i, bit))
        if diffs:
            i, bit = random.choice(diffs)
            if (v[i] >> bit) & 1:
                out[i] |= (1 << bit)
            else:
                out[i] &= ~(1 << bit)
    return bytes(out)

class AssocMemory:
    def __init__(self, centroids=None, codebook_size=256, alpha=0.02, threshold=25):
        self.alpha = alpha
        self.threshold = threshold  # ham > threshold → новый паттерн
        self.usage = Counter()
        if centroids:
            self.centroids = list(centroids)
            self.codebook_size = len(centroids)
        else:
            self.codebook_size = codebook_size
            self.centroids = None  # lazy init

    def _ensure_init(self, v: bytes):
        if self.centroids is None:
            # init одним центроидом = v
            self.centroids = [v] + [bytes(V_BYTES) for _ in range(self.codebook_size - 1)]

    def _nearest(self, v: bytes):
        best_idx = 0
        best_dist = hamming(v, self.centroids[0])
        for i in range(1, len(self.centroids)):
            d = hamming(v, self.centroids[i])
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, best_dist

    def encode(self, v: bytes) -> bytes:
        assert len(v) == V_BYTES, f"|V|={V_BYTES}"
        self._ensure_init(v)
        idx, _ = self._nearest(v)
        self.usage[idx] += 1
        # K = idx packed into K_BYTES (little endian)
        return struct.pack("<I", idx)[:K_BYTES]

    def decode(self, k: bytes, topk=10) -> list:
        assert len(k) == K_BYTES
        idx = struct.unpack("<I", k + b"\x00" * (4 - len(k)))[0] % len(self.centroids)
        # возвращаем окрестность: centroid + ближайшие центроиды по ham
        dists = [(hamming(self.centroids[idx], c), i) for i, c in enumerate(self.centroids)]
        dists.sort()
        return [self.centroids[i] for _, i in dists[:topk]]

    def update(self, v: bytes):
        """Online дообучение на новом V без полного переобучения."""
        self._ensure_init(v)
        idx, dist = self._nearest(v)
        self.usage[idx] += 1
        if dist > self.threshold:
            # новый паттерн — заменяем LRU код
            # находим наименее используемый
            lru = min(range(len(self.centroids)), key=lambda i: self.usage[i])
            # не заменяем если lru — это и есть nearest (чтобы не затереть активный)
            if lru == idx:
                # берем второй LRU
                sorted_usage = sorted(range(len(self.centroids)), key=lambda i: self.usage[i])
                lru = sorted_usage[1] if len(sorted_usage) > 1 else lru
            self.centroids[lru] = v
            self.usage[lru] = 1
        else:
            # EMA-обновление центроида
            self.centroids[idx] = majority_update(self.centroids[idx], v, self.alpha)

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"centroids": self.centroids, "usage": dict(self.usage),
                         "alpha": self.alpha, "threshold": self.threshold}, f)

    @staticmethod
    def load(path, alpha=None, threshold=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        # совместимость со старыми pkl где ключ "centroids"
        cents = d.get("centroids", d.get("centroids", []))
        mem = AssocMemory(centroids=cents, alpha=alpha or d.get("alpha", 0.02),
                          threshold=threshold or d.get("threshold", 25))
        mem.usage = Counter(d.get("usage", {}))
        return mem

    def eval_same_code(self, data_path):
        """Проверка: noisy V' попадает в тот же K что и V"""
        with open(data_path, "rb") as f:
            n, vb = struct.unpack("<II", f.read(8))
            hits = 0
            for _ in range(n):
                v = f.read(vb); vn = f.read(vb)
                if self.encode(v) == self.encode(vn):
                    hits += 1
            return hits / n if n else 0

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="AssocMemory online тест")
    ap.add_argument("--data", default="/tmp/universal-model/src/assoc/data/synth_clustered.bin")
    ap.add_argument("--codebook", default="/tmp/universal-model/src/assoc/transformer/codebook_clustered.pkl")
    ap.add_argument("--online-data", default=None, help="новый датасет для online дообучения")
    args = ap.parse_args()

    mem = AssocMemory.load(args.codebook)
    print(f"loaded {len(mem.centroids)} centroids, alpha={mem.alpha} threshold={mem.threshold}")
    base = mem.eval_same_code(args.data)
    print(f"base same-code recall: {base:.3f} (ожидается 1.0)")

    if args.online_data and os.path.exists(args.online_data):
        # дообучаем на новых данных
        import struct
        with open(args.online_data, "rb") as f:
            n, vb = struct.unpack("<II", f.read(8))
            for _ in range(n):
                v = f.read(vb); vn = f.read(vb)
                mem.update(v)
                mem.update(vn)
        print(f"after online update on {args.online_data}: recall={mem.eval_same_code(args.data):.3f}, new={mem.eval_same_code(args.online_data):.3f}")
        mem.save("/tmp/universal-model/src/assoc/transformer/codebook_online.pkl")
        print("saved codebook_online.pkl")
    else:
        # демо online: 10 новых случайных паттернов
        import random
        print("demo online: 10 новых V (раньше не видел)")
        for _ in range(10):
            v_new = bytes(random.getrandbits(8) for _ in range(V_BYTES))
            before = mem.encode(v_new)
            # 5 noisy копий
            for _ in range(5):
                vn = bytearray(v_new)
                for _ in range(2):
                    bi = random.randrange(V_BYTES)
                    vn[bi] ^= 1 << random.randrange(8)
                mem.update(bytes(vn))
            mem.update(v_new)
            after = mem.encode(v_new)
            print(f"  before={before.hex()} after={after.hex()} same={before==after} (LRU замена если dist>threshold)")
        print(f"recall after demo: {mem.eval_same_code(args.data):.3f}")
