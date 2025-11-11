# vild_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ViLDHead(nn.Module):
    """
    Student 모델의 region embedding을 텍스트 임베딩 공간과 동일한 차원으로 투영(projection)하는 헤드

    - CrossEntropyLoss 기반 분류용으로 cosine 정규화는 제거함
    - 구조: Linear -> LayerNorm -> ReLU
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        x = self.activation(x)
        return x  # 정규화 제거됨
    