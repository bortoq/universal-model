"""
Baseline tied-VQ для assoc: V(256 бит)->K(32 бит)->V'
Без трансформера v0.1, MLP + VQ. Интерфейс encode/decode из spec.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VQ(nn.Module):
    def __init__(self, codebook_size=1024, dim=32):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1/codebook_size, 1/codebook_size)
        self.codebook_size = codebook_size
        self.dim = dim

    def forward(self, z):
        # z: [B, dim]
        dist = torch.cdist(z, self.codebook.weight)  # [B, K]
        idx = dist.argmin(dim=1)  # [B]
        q = self.codebook(idx)
        # straight-through
        q_st = z + (q - z).detach()
        # losses
        commit_loss = F.mse_loss(q.detach(), z)
        codebook_loss = F.mse_loss(q, z.detach())
        return q_st, idx, commit_loss + codebook_loss

class AssocVQ(nn.Module):
    def __init__(self, v_bits=256, k_bits=32, hidden=128, codebook_size=1024):
        super().__init__()
        v_bytes = v_bits // 8
        k_bytes = k_bits // 8
        self.v_bits = v_bits
        self.k_bits = k_bits
        # V bytes -> hidden -> dim (k_bits)
        self.encoder = nn.Sequential(
            nn.Linear(v_bytes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k_bytes * 8)  # continuous before VQ, dim = k_bytes for simplicity use 32
        )
        # For VQ dim = 32 (k_bytes*8 bits -> treat as 32-dim float)
        self.vq_dim = 32
        self.enc_proj = nn.Linear(k_bytes*8, self.vq_dim)
        self.vq = VQ(codebook_size, self.vq_dim)
        self.dec_proj = nn.Linear(self.vq_dim, hidden)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, v_bytes),
        )
        self.v_bytes = v_bytes
        self.k_bytes = k_bytes

    def encode(self, v_bytes_tensor):
        # v: [B, v_bytes] uint8 -> float [0,1]
        x = v_bytes_tensor.float() / 255.0
        h = self.encoder(x)
        z = self.enc_proj(h)
        _, idx, _ = self.vq(z)
        # idx -> bytes (32 бит = 4 байта, codebook 1024 needs 10 bits -> pack into 4 bytes)
        # пока возвращаем idx как K (int)
        return idx

    def forward(self, v):
        x = v.float() / 255.0
        h = self.encoder(x)
        z = self.enc_proj(h)
        q, idx, vq_loss = self.vq(z)
        d = self.dec_proj(q)
        recon = self.decoder(d)  # [B, v_bytes] logits 0-255
        # recon loss: MSE in byte space
        recon_loss = F.mse_loss(recon, x * 255)
        return recon, idx, recon_loss, vq_loss

def hamming(a, b):
    # a,b: [N, 32] uint8
    import numpy as np
    return np.unpackbits(np.bitwise_xor(a, b), axis=1).sum(axis=1)
