"""Конфиг пирамиды Exp"""
# S0 = n * 2**n бит
N = 8  # n, тогда S0=8*256=2048 бит (256 байт) — демо
V_BYTES = 32  # |V| = 256 бит
K_BYTES = 4   # |K| = 32 бита, сжатие 8x
# t = ceil(log_{|V|/|K|}(S0/|K|))
import math
S0_BITS = N * (2 ** N)
S0_BYTES = S0_BITS // 8
RATIO = (V_BYTES * 8) // (K_BYTES * 8)  # |V|/|K| = 8
T = math.ceil(math.log(S0_BITS / (K_BYTES * 8), RATIO)) if S0_BITS > K_BYTES*8 else 1

def info():
    return f"n={N} S0={S0_BITS}b ({S0_BYTES}B) V={V_BYTES*8}b K={K_BYTES*8}b ratio={RATIO} t={T}"
