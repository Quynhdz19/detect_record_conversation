import torch.nn as nn


class lossAV(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = nn.Linear(256, 2)

    def forward(self, x, labels=None):
        x = x.squeeze(1)
        x = self.FC(x)
        return x[:, 1].reshape(-1).detach().cpu().numpy()
