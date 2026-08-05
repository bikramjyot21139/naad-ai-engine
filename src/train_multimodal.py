import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class NAADMultimodalEngine(nn.Module):
    def __init__(self, num_classes=5, audio_dim=32):
        super(NAADMultimodalEngine, self).__init__()
        self.text_backbone = AutoModel.from_pretrained(TOKENIZER_NAME)
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fusion_dim = 384 + 64
        
        self.classifier_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, audio_feats):
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = text_outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        text_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        audio_emb = self.audio_projection(audio_feats)
        fused = torch.cat((text_emb, audio_emb), dim=1)
        
        logits = self.classifier_head(fused)
        risk_score = self.regression_head(fused)
        return logits, risk_score.squeeze(-1)
