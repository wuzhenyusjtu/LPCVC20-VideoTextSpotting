import torch
from torch.autograd import Variable
import misc
import dataset
from PIL import Image
import time
import torch.nn as nn

import models.crnn as crnn


model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = CRNN(32, 1, 37, 256)
# if torch.cuda.is_available():
#     model = model.cuda()
model = model.cpu()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = misc.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
# if torch.cuda.is_available():
#     image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
t = time.time()
preds = model(image)
print("Time used:{} ".format(time.time() - t))

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))

dyn_q_model = torch.quantization.quantize_dynamic(
    model, {nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LSTM, nn.Linear}, dtype=torch.qint8
)

converter = misc.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
# if torch.cuda.is_available():
#     image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

dyn_q_model.eval()
t = time.time()
preds = dyn_q_model(image)
print("Time used:{} ".format(time.time() - t))

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))

