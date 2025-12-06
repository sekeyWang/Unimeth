import math
import torch
from torch import Tensor, nn


class Processor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_size//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=hidden_size//2),
            nn.GELU(),
            nn.Conv1d(in_channels=hidden_size//2, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.GELU()
        )
    
    def forward(self, x: Tensor):
        x = x.reshape(x.size()[0], 1, -1)
        x = self.process(x)
        x = x.permute(0, 2, 1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=260):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        self.register_buffer('encoding', encoding)

    def forward(self, x, pos):
        return x + self.encoding[pos]

class Threshold_based_loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, pos_ratio):
        n = logits.size(0)
        num_positive = int(torch.round(pos_ratio * n))
        sorted_logits, indices = torch.sort(logits, descending=True)
        pos_samples = sorted_logits[:num_positive]
        neg_samples = sorted_logits[num_positive:]
        if len(pos_samples) > 0:
            pos_loss = -torch.log(pos_samples)
        else:
            pos_loss = torch.tensor([], device=logits.device)
        if len(neg_samples) > 0:
            neg_loss = -torch.log(1 - neg_samples)
        else:
            neg_loss = torch.tensor([], device=logits.device)
        total_loss = torch.concat([pos_loss, neg_loss]).mean()
        return total_loss
