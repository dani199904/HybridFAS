"""
HybridFAS (SKT-Live V3)
Multi-Scale CNN + Dual-Token Transformer for Face Anti-Spoofing

Author: Danial Ahmed
"""



# ============================================================
# V2.5 — FULL EXPLICIT ARCHITECTURE (SINGLE CELL)
# ============================================================

import torch
from torch import nn
import torch.nn.functional as F

# -------------------- Patch Embedding --------------------
class PatchEmbedV2_3(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, embed_dim, kernel_size=16, stride=16)

    def forward(self, x):
        return self.conv(x)  # [B, 256, 14, 14]


# -------------------- Spoof Branch Sub-CNNs --------------------
class SpoofEdgeCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.b2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))


class SpoofReflectCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))


class SpoofNoiseCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.b2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))


class SpoofBranchV2_3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge = SpoofEdgeCNN(dim)
        self.reflect = SpoofReflectCNN(dim)
        self.noise = SpoofNoiseCNN(dim)

    def forward(self, x):
        return (self.edge(x) + self.reflect(x) + self.noise(x)) / 3


# -------------------- Real Branch Sub-CNNs --------------------
class RealSkinCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))


class RealIllumCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 7, padding=3)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 7, padding=3)
        self.b2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))


class RealStructureCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))


class RealBranchV2_3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.skin = RealSkinCNN(dim)
        self.illum = RealIllumCNN(dim)
        self.struct = RealStructureCNN(dim)

    def forward(self, x):
        return (self.skin(x) + self.illum(x) + self.struct(x)) / 3


# -------------------- Learnable Fusion Gate --------------------
class FusionGateV2_3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim * 3, dim, 1)
        self.c2 = nn.Conv2d(dim, 3, 1)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, base, spoof, real):
        x = torch.cat([base, spoof, real], dim=1)
        x = F.relu(self.c1(x))
        x = self.drop(x)
        w = F.softmax(self.c2(x), dim=1)
        return w[:, 0:1] * base + w[:, 1:2] * spoof + w[:, 2:3] * real


# -------------------- Transformer Encoder --------------------
class TransformerEncoderV2_3(nn.Module):
    def __init__(self, dim=256, layers=4, heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        return self.encoder(x)


# --------------------  V2.3 --------------------
class SKTLiveV2_5(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.patch = PatchEmbedV2_3(3, embed_dim)
        self.spoof = SpoofBranchV2_3(embed_dim)
        self.real  = RealBranchV2_3(embed_dim)
        self.fusion = FusionGateV2_3(embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.transformer = TransformerEncoderV2_3(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(), # Upgraded to GELU for smoother gradients
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        base = self.patch(x)
        s_feat = self.spoof(base)
        r_feat = self.real(base)
        
        fused = self.fusion(base, s_feat, r_feat)
        fused = F.gelu(fused) 
        
        tokens = fused.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        
        B = tokens.size(0)
        cls_token = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        
        out = self.transformer(tokens)
        logits = self.head(out[:, 0])
        return logits

