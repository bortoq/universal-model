import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from exp.pyramid import encode_pyramid, decode_pyramid, encode_level, decode_level, hamming
from exp.config import V_BYTES, S0_BYTES
from assoc.transformer.assoc_memory import AssocMemory
import random

def test_single_level():
    mem = AssocMemory.load(os.path.join(os.path.dirname(__file__), "../../assoc/transformer/codebook_clustered.pkl"))
    # берем V из обучающего датасета
    import struct
    data = os.path.join(os.path.dirname(__file__), "../../assoc/data/synth_clustered.bin")
    with open(data, "rb") as f:
        n, vb = struct.unpack("<II", f.read(8))
        v = f.read(vb); vn = f.read(vb)
        s0 = v
    levels = encode_pyramid(s0, mem)
    assert len(levels) == 2  # S0 32B -> 4B
    rec = decode_pyramid(levels, mem)
    err = hamming(s0, rec)
    print(f"single V ham {err}/256 {err/256*100:.1f}%")
    assert err <= 70  # clustered worst p95 59, allow 70

def test_l1():
    mem = AssocMemory.load(os.path.join(os.path.dirname(__file__), "../../assoc/transformer/codebook_clustered.pkl"))
    import struct
    data = os.path.join(os.path.dirname(__file__), "../../assoc/data/synth_clustered.bin")
    with open(data, "rb") as f:
        n, vb = struct.unpack("<II", f.read(8))
        # собираем S0 из 8 последовательных V из датасета
        vs = []
        for _ in range(8):
            v = f.read(vb); vn = f.read(vb)
            vs.append(v)
        s0 = b"".join(vs)
    levels = encode_pyramid(s0, mem)
    s1 = levels[1]
    s0_rec = decode_level(s1, mem)
    err = hamming(s0, s0_rec[:len(s0)])
    print(f"L1 ham {err}/{len(s0)*8} {err/(len(s0)*8)*100:.1f}%")
    assert err < 200  # ~4.5% expected <10%

if __name__ == "__main__":
    test_single_level()
    test_l1()
    print("exp tests passed")
