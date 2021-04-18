import torch
import torch.nn as nn


class FOTS_r(nn.Module):
    def __init__(self, part1_path):
        super().__init__()
        self.part1 = torch.jit.load(part1_path)

    def forward(self, x):
        e1, e2, e3 = self.part1(x)
        return e1, e2, e3
