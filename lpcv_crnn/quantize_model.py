import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import time
import cv2
import os
import nltk
import argparse
from models.quantized_crnn import CRNN_q

# New load fucntion for CRNN
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

# Get CRNN results by giving a directory containing images/data.
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
#         print('{} => {}, gt label:{}, e_dist:{}'.format(raw_pred, sim_pred, label, edit_distance))
        total_dist += edit_distance
        total_cnt += 1
        
    return total_dist, total_cnt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pruned', required=True, help="path to pruned model (to continue training)")
    parser.add_argument('--expr_dir', required=True, help='Where to store samples and models')
#     parser.add_argument('--imagedir', required=True, help='path to sample datasets')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    # TODO(meijieru): epoch -> iter
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--qconfig', default='fbgemm', help="type of quantization configure (fbgemm or qnnpack)")
    parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
    parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
    opt = parser.parse_args()
    opt.cuda = True
    opt.adadelta = True
    print(opt)

    model_path = opt.pruned
    img_path = './data/demo.png'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

    if opt.qconfig not in ['fbgemm', 'qnnpack']:
        raise NameError('qconfig parameter should be qnnpack or fbgemm')

    model = CRNN_q(32, 1, 37, 256)
    model.eval()
    model = model.cpu()
    model.load_state_dict(load_multi(model_path), strict=False)

    for idx, m in enumerate(model.rnn.children()):
        m.set_wrap()

    model.fuse_model()
    print(model)

    qconf = torch.quantization.get_default_qconfig(opt.qconfig)
    model.cnn.qconfig = qconf
    for idx, m in enumerate(model.rnn.children()):
        m.embedding_w.qconfig = qconf

    torch.quantization.prepare(model.cnn, inplace=True)
    for idx, m in enumerate(model.rnn.children()):
        torch.quantization.prepare(m.embedding_w, inplace=True)

    print(model)

    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((100, 32))
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = transformer(image)
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s => %-20s' % (raw_pred, sim_pred))

#     image_dir = opt.imagedir
#     get_crnn_res(model, image_dir, converter, transformer)

    torch.quantization.convert(model.cnn, inplace=True)
    for idx, m in enumerate(model.rnn.children()):
        torch.quantization.convert(m.embedding_w, inplace=True)
    print(model)
    torch.jit.save(torch.jit.script(model.cnn), '{}/cnn_CRNN_{}.torchscript'.format(opt.expr_dir, opt.qconfig))
    print("cnn part is saved to {}/cnn_CRNN_{}.torchscript".format(opt.expr_dir, opt.qconfig))
    for idx, m in enumerate(model.rnn.children()):
        torch.jit.save(torch.jit.script(m.embedding_w), '{}/embd_w{}_CRNN_{}.torchscript'.format(opt.expr_dir, idx, opt.qconfig))
        print("embedding part {} is saved to {}/embd_w{}_CRNN_{}.torchscript".format(idx,opt.expr_dir, idx, opt.qconfig ))
