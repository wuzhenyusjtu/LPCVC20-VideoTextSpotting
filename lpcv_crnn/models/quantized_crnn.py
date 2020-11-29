import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torch
import copy
from models.pruned_crnn import CRNN_pruned

class RNN_embedding(nn.Module):
    def __init__(self, nHidden, nOut):
        super(RNN_embedding, self).__init__()
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, input):
        input = self.quant(input)
        output = self.embedding(input)
        output = self.dequant(output)
        return output
    
# Specific for quantized model
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        
        self.embedding_w = RNN_embedding(nHidden, nOut)
    
    def set_wrap(self):
        self.embedding_w.embedding = copy.deepcopy(self.embedding)
    
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding_w(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output
    
class CRNN_q(CRNN_pruned):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN_q, self).__init__(imgH, nc, nclass, nh, n_rnn, leakyRelu)
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(self.nm[-1], nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.cnn.quant = QuantStub()
        self.cnn.dequant = DeQuantStub()
        
    def forward(self, input):
        # conv features
        input = self.cnn.quant(input)
        #print(input.dtype)
        for idx, m in enumerate(self.cnn.children()):
            if idx > 20:
                break
            input = m(input)
        # conv = self.cnn(input)
        conv = self.cnn.dequant(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        
        return output
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self.cnn, [['conv0', 'relu0'], ['conv1', 'relu1'], ['conv2', 'batchnorm2', 'relu2'],\
                                              ['conv3', 'relu3'], ['conv4', 'batchnorm4', 'relu4'], ['conv5', 'relu5'],\
                                              ['conv6', 'batchnorm6', 'relu6']], inplace=True)
