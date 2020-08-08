import cv2
import numpy as np
import torch
from torch.autograd import Variable


def np_to_variable(x, is_cuda=False, dtype=torch.FloatTensor):
  v = torch.from_numpy(x).type(dtype)
  if is_cuda:
      v = v.cuda()
  return v


def load_net(fname, net, optimizer=None):
  sp = torch.load(fname)
  step = sp['step']
  try:
    learning_rate = sp['learning_rate']
  except:
    import traceback
    traceback.print_exc()
    learning_rate = 0.001
  opt_state = sp['optimizer']
  sp = sp['state_dict']
  for k, v in net.state_dict().items():
    try:
      param = sp[k]
      v.copy_(param)
    except:
      import traceback
      traceback.print_exc()

  if optimizer is not None:
    try:
      optimizer.load_state_dict(opt_state)
    except:
      import traceback
      traceback.print_exc()

  print(fname)
  return step, learning_rate


def resize_image(image, max_size=1585152, scale_up=True):
    if scale_up:
        image_size = [image.shape[1] * 3 // 32 * 32, image.shape[0] * 3 // 32 * 32]
    else:
        image_size = [image.shape[1] // 32 * 32, image.shape[0] // 32 * 32]
    while image_size[0] * image_size[1] > max_size:
        image_size[0] /= 1.2
        image_size[1] /= 1.2
        image_size[0] = int(image_size[0] // 32) * 32
        image_size[1] = int(image_size[1] // 32) * 32

    resize_h = int(image_size[1])
    resize_w = int(image_size[0])

    scaled = cv2.resize(image, dsize=(resize_w, resize_h))
    return scaled, (resize_h, resize_w)
