import torch
import torch.nn as nn

class FOTSModel_q(nn.Module):
    def __init__(self, part1_path, part2_path, part3_path, rejector_path, fc_path, thrshold=0.3):
        super().__init__()
        self.fc = nn.Linear(128, 1)
        self.fc.load_state_dict(torch.load(fc_path))
        torch.quantization.quantize_dynamic(self, {nn.Linear}, dtype=torch.qint8, inplace=True)
        print(self.fc)
        
        self.part1 = torch.jit.load(part1_path)
        self.part2 = torch.jit.load(part2_path)
        self.part3 = torch.jit.load(part3_path)
        self.rejector = torch.jit.load(rejector_path)
        
        self.sigm = nn.Sigmoid()
        
        self.thrshold = thrshold
        
    def forward(self, x):
        e1 = self.part1(x)
        e2 = self.part2(e1)
        x = self.rejector(e2)
        x = self.fc(x)
        x = self.sigm(x)
        if x < self.thrshold:
            return None, None, None
        confidence, distances, angle = self.part3(e1, e2)
        return confidence, distances, angle