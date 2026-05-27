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

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.encoder = GPT2Model.from_pretrained("gpt2")
        self.fc = nn.Linear(768, num_classes)
        self.encoder.requires_grad_(False)

    def forward(self, input_ids):
        output = self.encoder(input_ids=input_ids)
        last_hidden = output.last_hidden_state[:, -1, :]
        logits = self.fc(last_hidden)
        return logits
