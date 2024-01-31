import torch
from torch import nn
import torch.nn.functional as F
from .base_model import ConvAutoregressiveBaseModel, CHANNEL_DIM

class DilatedCausalConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, dilation_factor, **kwargs):
        kwargs.setdefault('kernel_size', 2)
        kwargs.setdefault('stride', 1)
        padding = kwargs['kernel_size'] - 1 + dilation_factor - 1
        self.truncation = padding
        kwargs.setdefault('padding', padding)
        super().__init__(in_channels, out_channels, dilation=dilation_factor, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        return x[..., :-self.truncation]
    
class WaveNetBlock(nn.Module):

    def __init__(self, block_channels, skip_channels, dilation_factor, config):
        super().__init__()
        assert skip_channels == block_channels or config.get('use_res_conv')
        self.block_channels = block_channels
        self.dcc = DilatedCausalConv1d(block_channels, block_channels*2, dilation_factor=dilation_factor, bias=config['bias']) # filter conv
        self.skip_conv = nn.Conv1d(block_channels, skip_channels, 1, bias=config['bias'])
        self.dropout = nn.Dropout(config['dropout'])
        if config.get('use_res_conv'):
            self.res_conv = nn.Conv1d(block_channels, block_channels, 1, bias=config['bias'])
        else:
            self.res_conv = lambda x: x
    
    def forward(self, x):
        residual = x
        x_act, x_gate = self.dcc(x).split((self.block_channels, self.block_channels), dim=CHANNEL_DIM)
        x = F.tanh(x_act) * F.sigmoid(x_gate)
        skip_out = self.skip_conv(x)
        block_out = self.res_conv(x)
        x = block_out + residual
        x = self.dropout(x)
        return x, skip_out

class WaveNet(ConvAutoregressiveBaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.dim = config['hidden_channels']
        self.skip_dim = config.get('skip_channels', self.dim)
        self.out_dim = config['out_channels']
        self.in_transform = nn.Conv1d(config['in_channels'], self.dim, 1)   # 1 x 1 convolution
        self.in_dropout = nn.Dropout(config['input_dropout'])
        self.condition_on_future = config.get('condition_on_future', False)
        self.has_dynamic_bias = config.get('dynamic_bias_layer') is not None
        self.is_probabilistic = config.get('is_probabilistic', False)
        
        self.blocks = nn.ModuleList([])

        for b in range(config['num_layers']):
            self.blocks.append(
                WaveNetBlock(self.dim, self.skip_dim, dilation_factor=2**b, config=config),
            )

        mlp_input_dim = self.skip_dim 

        if self.condition_on_future:
            mlp_input_dim += config['num_future_conditionals']

        self.pre_future_ops = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config['output_dropout']),
        )
        self.out_mlp = nn.Sequential(
            nn.Conv1d(mlp_input_dim, self.skip_dim, 1, bias=config['bias']), # 1 x 1 convolution
            nn.ReLU(),
            nn.Dropout(config['output_dropout']),
        )
        self.out_linear = nn.Conv1d(self.skip_dim, self.out_dim, 1, bias=True)
        
    def forward(self, x, future_conditionals=None):
        x = self.in_transform(x)
        x = self.in_dropout(x)
        x_skip = torch.zeros([*x.shape[:-2], self.skip_dim, x.shape[-1]], device=x.device)
        
        for block in self.blocks:
            x, skip_out = block(x)
            x_skip = x_skip + skip_out

        x_skip = self.pre_future_ops(x_skip)
        
        if self.condition_on_future:
            assert future_conditionals is not None
            x_skip = torch.cat((x_skip, future_conditionals), dim=CHANNEL_DIM)

        output = self.out_mlp(x_skip)
        estimate = self.out_linear(output)
        
        return estimate