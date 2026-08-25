#!/usr/bin/env python3
"""
Pyramid Exp: AoP + AoV -> exp-пространство
S0 (AoP-значения, n·2ⁿ) --AoV V->K--> S1 --V->K--> ... -> 1 ключ
Обратно: K -> окрестность (AoV decode) -> сборка AoP

pure-python, зависит от src/assoc/transformer/assoc_memory.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assoc.transformer.assoc_memory import AssocMemory
from exp.config import V_BYTES, K_BYTES, S0_BYTES, T
import struct

def hamming(a: bytes, b: bytes) -> int:
    return bin(int.from_bytes(bytes(x^y for x,y in zip(a,b)), 'big')).count('1')

def encode_level(data: bytes, mem: AssocMemory) -> bytes:
    """Один уровень пирамиды: режем data на V-окна -> K"""
    out = bytearray()
    for i in range(0, len(data), V_BYTES):
        chunk = data[i:i+V_BYTES]
        if len(chunk) < V_BYTES:
            chunk = chunk + b"\x00" * (V_BYTES - len(chunk))
        k = mem.encode(chunk)
        out.extend(k)
    return bytes(out)

def decode_level(keys: bytes, mem: AssocMemory) -> bytes:
    """Обратный уровень: K -> V (top1 centroid)"""
    out = bytearray()
    for i in range(0, len(keys), K_BYTES):
        k = keys[i:i+K_BYTES]
        if len(k) < K_BYTES:
            k = k + b"\x00" * (K_BYTES - len(k))
        # decode возвращает список, берем первый (centroid)
        v = mem.decode(k, topk=1)[0]
        out.extend(v)
    return bytes(out)

def encode_pyramid(s0: bytes, mem: AssocMemory):
    """S0 -> [S0, S1, ..., St]"""
    levels = [s0]
    cur = s0
    for _ in range(T):
        if len(cur) <= K_BYTES:
            break
        nxt = encode_level(cur, mem)
        levels.append(nxt)
        cur = nxt
        if len(cur) <= K_BYTES:
            break
    return levels

def decode_pyramid(levels, mem: AssocMemory):
    """Восстановление S0' из вершины пирамиды (последний уровень)"""
    # идем снизу вверх: St -> St-1 -> ... -> S0
    cur = levels[-1]
    # нам нужно знать исходные размеры каждого уровня для обрезки padding
    for lvl in reversed(levels[:-1]):
        decoded = decode_level(cur, mem)
        # обрезаем до размера lvl
        cur = decoded[:len(lvl)]
    return cur

def demo():
    import random
    print(f"pyramid config: {__import__('exp.config', fromlist=['info']).info()}")
    # Демо 1: один уровень (S0 = V) — показывает работу AoV без второго уровня
    random.seed(0)
    bases = [bytes(random.getrandbits(8) for _ in range(V_BYTES)) for _ in range(8)]
    # S0 как один V (32B) — t=1, чисто AoV
    s0_single = random.choice(bases)
    import os
    pkl = os.path.join(os.path.dirname(__file__), "../assoc/transformer/codebook_clustered.pkl")
    mem = AssocMemory.load(pkl)
    levels_single = encode_pyramid(s0_single, mem)
    print(f"[single] S0 {len(s0_single)}B -> levels {[len(l) for l in levels_single]}")
    rec_single = decode_pyramid(levels_single, mem)
    print(f"  single recovery ham {hamming(s0_single, rec_single)}/{V_BYTES*8} = {hamming(s0_single, rec_single)/V_BYTES/8*100:.2f}% (ожидается ~6.3/256=2.4%)")
    # Демо 2: полная пирамида S0=256B (8×V) -> S1 32B -> S2 4B (t=2)
    s0 = b"".join(random.choice(bases) for _ in range(S0_BYTES // V_BYTES))
    levels = encode_pyramid(s0, mem)
    print(f"[full] S0 {len(s0)}B -> levels {[len(l) for l in levels]} t={T}")
    for i, lvl in enumerate(levels):
        print(f"  S{i} {len(lvl)}B = {len(lvl)*8}b")
    # ошибка первого уровня (S0->S1->S0) — должна быть мала (~6*8=48 бит)
    s1 = levels[1]
    s0_from_s1 = decode_level(s1, mem)
    err_l1 = hamming(s0, s0_from_s1[:len(s0)])
    print(f"  L1 only (S1->S0) ham {err_l1}/{len(s0)*8} = {err_l1/(len(s0)*8)*100:.2f}% (ожидается ~2.4%)")
    # полная пирамида через S2 — S1 состоит из K (4B каждый), не из V (32B баз), поэтому out-of-distribution
    s0_rec = decode_pyramid(levels, mem)
    err = hamming(s0, s0_rec)
    print(f"  full pyramid (S2->S1->S0) ham {err}/{len(s0)*8} = {err/(len(s0)*8)*100:.2f}% — высокий из-за OOD S1 (K-конкатенация, не V)")
    print(f"  → нужен отдельный AoV кодбук для уровня S1 (обучить на конкатенациях K), тогда ошибка вернется к ~2%")
    # noisy
    s0_noisy = bytearray(s0)
    for _ in range(5):
        bi = random.randrange(len(s0_noisy))
        s0_noisy[bi] ^= 1 << random.randrange(8)
    s0_noisy = bytes(s0_noisy)
    levels_n = encode_pyramid(s0_noisy, mem)
    print(f"  noisy same top key? {levels[-1].hex() == levels_n[-1].hex()} (same-code) - пока False из-за OOD")

if __name__ == "__main__":
    demo()
