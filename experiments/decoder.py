import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBnRelu(nn.Module):
    '''[FC => BN => ReLU]'''
    def __init__(self, in_features, out_features):
        super(LinearBnRelu, self).__init__()
        self.inner_module = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.inner_module(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Decoder, self).__init__()
        self.linear1 = LinearBnRelu(in_features, 500)
        self.linear4 = LinearBnRelu(500, 4000)
        self.linear_out = nn.Linear(4000, out_features)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear4(x)
        x = self.linear_out(x)
        out = F.sigmoid(x)
        
        return out
