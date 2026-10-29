# -*- coding: utf-8 -*-
"""PyTorch encoder: range-Doppler-angle clip -> K tokens in LLM space.

Input  : float32 [B, T, 2, R, D, A]   (R=256, D=20, A=16)
Output : float32 [B, K, llm_dim]     (K=16, llm_dim=4096 for Qwen3-8B)

Design:
  - per-frame 3D CNN over (range, Doppler, angle)
  - adaptive pool -> 16 spatial tokens per frame
  - temporal transformer over T frames
  - K learnable queries compress T*16 tokens
  - 2-layer MLP projects to the LLM embedding width
"""

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    def __init__(self, in_ch=2, width=128, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=5, stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv3d(64, width, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((4, 2, 2))
        self.proj = nn.Linear(width, d_model)

    def forward(self, x):
        # x: [B*T, 2, R, D, A]
        h = self.net(x)
        h = self.pool(h)                 # [B*T, width, 4, 2, 2]
        h = h.flatten(2).transpose(1, 2)  # [B*T, 16, width]
        return self.proj(h)              # [B*T, 16, d_model]


class RadarIQEncoder(nn.Module):
    def __init__(self, t_frames=16, in_ch=2, d_model=256, n_queries=16,
                 n_heads=8, n_layers=2, llm_dim=4096):
        super().__init__()
        self.t_frames = t_frames
        self.d_model = d_model
        self.n_queries = n_queries

        self.frame = FrameEncoder(in_ch=in_ch, d_model=d_model)
        self.frame_pos = nn.Parameter(torch.zeros(1, t_frames, 1, d_model))
        nn.init.normal_(self.frame_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=1024,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.queries = nn.Parameter(torch.zeros(1, n_queries, d_model))
        nn.init.normal_(self.queries, std=0.02)
        self.qattn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        self.projector = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.GELU(),
            nn.Linear(1024, llm_dim),
        )

    def forward(self, x):
        # x: [B, T, 2, R, D, A]
        b, t, c, r, d, a = x.shape
        h = self.frame(x.reshape(b * t, c, r, d, a))  # [B*T, 16, d_model]
        h = h.reshape(b, t, 16, self.d_model)
        h = h + self.frame_pos[:, :t]
        h = h.reshape(b, t * 16, self.d_model)
        h = self.temporal(h)

        q = self.queries.expand(b, -1, -1)
        h, _ = self.qattn(q, h, h)
        h = self.norm(h)
        return self.projector(h)  # [B, K, llm_dim]


def build_encoder(llm_dim=4096, t_frames=16):
    return RadarIQEncoder(t_frames=t_frames, llm_dim=llm_dim)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
