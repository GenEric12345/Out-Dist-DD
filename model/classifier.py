import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from transformers import GPT2Model, GPT2TokenizerFast

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    get_bias_dropout_add_scale,
    modulate_fused,
)

import math

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(max_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.max_dim = max_dim

    def forward(self, t):
        # sinusoidal embedding, same as used in diffusion models
        half = self.max_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]  # (batch, half)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, max_dim)
        return self.mlp(embedding)  # (batch, hidden_dim)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.encoder = GPT2Model.from_pretrained("gpt2")
        self.timestep_emb = TimestepEmbedding(hidden_dim = 768)
        self.fc = nn.Linear(768, num_classes)
        self.encoder.requires_grad_(False)

    def forward(self, input_ids, t):
        output = self.encoder(input_ids=input_ids)
        last_hidden = output.last_hidden_state[:, -1, :]
        t_emb = self.timestep_emb(t)
        logits = self.fc(last_hidden+t_emb)
        return logits
