import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import time
import cv2
import os
import nltk

import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torch
import copy

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


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
#         nm = [64, 128, 256, 256, 512, 512, 512]
        nm = [64, 128, 128, 77, 154, 77, 77]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(nm[-1], nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        
        self.cnn.quant = QuantStub()
        self.cnn.dequant = DeQuantStub()
        
    def forward(self, input):
        # conv features
        input = self.cnn.quant(input)
        print(input.dtype)
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

# model_path = '/data/yunhe/nni_crnn_finetune_pruned_l1_multi3_sample_all/netCRNN_1_100.pth'
model_path = '/data/yunhe/nni_crnn_finetune_pruned_l1_multi3_sample_all_15/netCRNN_0.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

def load_multi(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        if 'module.' in name:
            name = name.replace('module.', '')
#         name = k[7:17] + k[24:]
        new_state_dict[name] = v
    return new_state_dict

model = CRNN(32, 1, 37, 256)
model.eval()
# print(model)
model = model.cpu()
# print('loading pretrained model from %s' % model_path)
# model.load_state_dict(torch.load(model_path))
model.load_state_dict(load_multi(model_path), strict=False)

for idx, m in enumerate(model.rnn.children()):
    m.set_wrap()

model.fuse_model()
print(model)

qconf = torch.quantization.get_default_qconfig('fbgemm')
model.cnn.qconfig = qconf
for idx, m in enumerate(model.rnn.children()):
    m.embedding_w.qconfig = qconf
# model.rnn.qconfig = qconf
#     model.proposal_generator.fcos_head.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model.cnn, inplace=True)
for idx, m in enumerate(model.rnn.children()):
    torch.quantization.prepare(m.embedding_w, inplace=True)
# torch.quantization.prepare(model.rnn, inplace=True)

print(model)

converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32))
image = cv2.imread(img_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = Image.open(img_path).convert('L')
image = transformer(image)
# if torch.cuda.is_available():
#     image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))

def get_crnn_res(model, image_dir, converter, transformer):
    total_dist = 0
    total_cnt = 0
    trans_table = dict.fromkeys(map(ord, '!*^&$@\',.:;-'), None)
    for img_name in os.listdir(image_dir)[0:100]:
        if not img_name.endswith('.jpg'):
            continue
        img_path = os.path.join(image_dir, img_name)
        
        label = str(img_name.split(':')[0]).translate(trans_table)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        text_patch = transformer(image)
#         if torch.cuda.is_available():
#             text_patch = text_patch.cuda()
        text_patch = text_patch.view(1, *text_patch.size())
        text_patch = Variable(text_patch)  
        #     print(text_patch.shape)
        preds = model(text_patch)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        edit_distance = nltk.edit_distance(sim_pred.upper(), label.upper())
        print('{} => {}, gt label:{}, e_dist:{}'.format(raw_pred, sim_pred, label, edit_distance))
        total_dist += edit_distance
        total_cnt += 1
        
    return total_dist, total_cnt

image_dir = './train_dataset_bezier'
get_crnn_res(model, image_dir, converter, transformer)

torch.quantization.convert(model.cnn, inplace=True)
for idx, m in enumerate(model.rnn.children()):
    torch.quantization.convert(m.embedding_w, inplace=True)
#     torch.quantization.convert(m.quant, inplace=True)
print(model)
torch.jit.save(torch.jit.script(model.cnn), './cnn_final_15_qnnpack.torchscript')
for idx, m in enumerate(model.rnn.children()):
    torch.jit.save(torch.jit.script(m.embedding_w), './crnn_embd_w{}_final_15_qnnpack.torchscript'.format(idx))
#     torch.jit.save(torch.jit.script(m.quant), './test_crnn_all_quant.torchscript')
