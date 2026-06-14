import math
import torch
from torch import Tensor, nn


class SignalProcessor(nn.Module):
    """
    CNN Signal Processor for UniMeth.
    
    Args:
        config: Model configuration dict containing:
            - d_model: Output hidden dimension
            - num_cnn_layers: Number of CNN layers
            - kernel_sizes: List of kernel sizes for each layer
            - strides: List of strides for each layer
    """
    def __init__(self, config):
        super().__init__()
        
        hidden_size = config['d_model']
        num_layers = config.get('num_cnn_layers', 2)
        kernel_sizes = config.get('kernel_sizes', [3] * num_layers)
        strides = config.get('strides', [2] * num_layers)
        
        layers = []
        in_ch = 1
        
        for i in range(num_layers):
            out_ch = hidden_size if i == num_layers - 1 else hidden_size // 2
            k = kernel_sizes[i]
            s = strides[i]
            p = (k - 1) // 2
            
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
                nn.BatchNorm1d(out_ch),
                nn.GELU()
            ])
            in_ch = out_ch
        
        self.process = nn.Sequential(*layers)
        self.total_stride = math.prod(strides)
        
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


