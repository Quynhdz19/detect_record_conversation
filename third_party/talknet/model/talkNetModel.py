import torch
import torch.nn as nn

from .attentionLayer import attentionLayer
from .audioEncoder import audioEncoder
from .visualEncoder import visualConv1D, visualFrontend, visualTCN


class talkNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visualFrontend = visualFrontend()
        self.visualTCN = visualTCN()
        self.visualConv1D = visualConv1D()
        self.audioEncoder = audioEncoder(layers=[3, 4, 6, 3], num_filters=[16, 32, 64, 128])
        self.crossA2V = attentionLayer(d_model=128, nhead=8)
        self.crossV2A = attentionLayer(d_model=128, nhead=8)
        self.selfAV = attentionLayer(d_model=256, nhead=8)

    def forward_visual_frontend(self, x):
        b, t, w, h = x.shape
        x = x.view(b * t, 1, 1, w, h)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(b, t, 512)
        x = x.transpose(1, 2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        return x.transpose(1, 2)

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        return self.audioEncoder(x)

    def forward_cross_attention(self, x1, x2):
        return self.crossA2V(src=x1, tar=x2), self.crossV2A(src=x2, tar=x1)

    def forward_audio_visual_backend(self, x1, x2):
        x = torch.cat((x1, x2), 2)
        x = self.selfAV(src=x, tar=x)
        return torch.reshape(x, (-1, 256))
