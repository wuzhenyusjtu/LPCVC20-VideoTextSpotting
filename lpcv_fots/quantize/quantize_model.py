# Code for l1 0.5
import sys
import os
import argparse
import cv2
import numpy as np
import torch
import torch.utils.data
import tqdm
import torch.quantization._numeric_suite as ns
from collections import OrderedDict
import copy

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
print(parentdir)
sys.path.append(parentdir)

from models.quanted_fots import FOTS_quanted
from modules.parse_polys import parse_polys


# Rename state dict keys to match quantized models architectures
def rename_state_dict_keys(model_path):
    state_dict = model_path['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == 'resnet':
            name = k
            new_state_dict[name] = v
        elif k[:5] == 'conv1' or k[:8] == 'encoder1' or k[:8] == 'encoder2' or k[:8] == 'encoder3':
            name = k
            new_state_dict[name] = v
            name = 'part1.' + k
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
            name = 'part2.' + k
            new_state_dict[name] = v
    return new_state_dict


def dsample_image(img, ksize):
    h, w = img.shape[:2]
    resized_img = np.lib.stride_tricks.as_strided(
        img,
        shape=(int(h / ksize), int(w / ksize), ksize, ksize, 3),
        strides=img.itemsize * np.array([ksize * w * 3, ksize * 3, w * 3, 1 * 3, 1]))
    return resized_img[:, :, 0, 0].copy()


# Compute error for each layer
def compute_error(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x-y)
    return 20*torch.log10(Ps/Pn)


# Generate image tensor for model input
def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if h == 2160 and w == 3840:
        k_size = 4
    elif h == 1080 and w == 1920:
        k_size = 2
    else:
        k_size = 1  # Just for robustness
    # Down sample image
    img = dsample_image(img, k_size)

    scale_y = 270 / min(img.shape[0], img.shape[1])  # 1248 # 704
    scaled_image = cv2.resize(img, dsize=(0, 0), fx=scale_y, fy=scale_y, interpolation=cv2.INTER_CUBIC)

    scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
    scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()
    return img_tensor, scale_y


# Parse bounding boxes from model outputs
def post_process(confidence, distances, angle, scale):
    if confidence is None:
        return []
    confidence = torch.sigmoid(confidence).squeeze().data.cpu().numpy()
    distances = distances.squeeze().data.cpu().numpy()
    angle = angle.squeeze().data.cpu().numpy()
    polys = parse_polys(confidence, distances, angle, 0.95, 0.3)  # , img=orig_scaled_image)

    reshaped_pred_polys = []
    for id in range(polys.shape[0]):
        reshaped_pred_polys.append(np.array(
            [int(polys[id, 0] / scale), int(polys[id, 1] / scale), int(polys[id, 2] / scale),
             int(polys[id, 3] / scale),
             int(polys[id, 4] / scale), int(polys[id, 5] / scale), int(polys[id, 6] / scale),
             int(polys[id, 7] / scale)]).reshape((4, 2)))
    return reshaped_pred_polys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate-folder', type=str, required=True,
                        help='Path to folder with images for calibrating')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--pretrain-model', type=str, default=None, help='Pretrained model name in save dir')
    parser.add_argument('--backends', type=str, default='fbgemm', help='Quantization backends')
    args = parser.parse_args()

    images_folder = args.calibrate_folder
    pbar = tqdm.tqdm(os.listdir(images_folder), desc='Test', ncols=80)

    # Load model state dict
    net = FOTS_quanted()
    checkpoint_name = args.pretrain_model
    checkpoint = torch.load(checkpoint_name, map_location='cpu')
    new_state_dict = rename_state_dict_keys(checkpoint)

    net.load_state_dict(new_state_dict, strict=True)
    _dummy_input_data = torch.rand(1, 3, 299, 299)

    # fp_net for floating point reference
    fp_net = copy.deepcopy(net)

    # Must set eval for Static Post Quantization
    net.eval()
    fp_net.eval()

    # Get quantization configure for post training quantization
    qcf = torch.quantization.get_default_qconfig(args.backends)

    # Set quantization configure to all parts in FOTs model
    net.part1.qconfig = qcf
    net.part2.qconfig = qcf

    # Fuse all parts in the FOTs model
    net.fuse_model()

    # Prepare the quantization for all parts
    torch.quantization.prepare(net.part1, inplace=True)
    torch.quantization.prepare(net.part2, inplace=True)

    # Use some images from the folder for evaluation or calibration
    for image_name in pbar:
        image = cv2.imread(os.path.join(images_folder, image_name), cv2.IMREAD_COLOR)
        image_tensor, scale = pre_process(image)
        confidence, distances, angle = net(image_tensor)
        boxes = post_process(confidence, distances, angle, scale)

    # Convert all parts into quantized ones
    quantized_part1 = torch.quantization.convert(net.part1, inplace=True)
    quantized_part2 = torch.quantization.convert(net.part2, inplace=True)

    # Compare each layer of part 1 and 2, the higher score the better
    wt_compare_dict_1 = ns.compare_weights(fp_net.part1.state_dict(), quantized_part1.state_dict())
    wt_compare_dict_2 = ns.compare_weights(fp_net.part2.state_dict(), quantized_part2.state_dict())

    for key in wt_compare_dict_1:
        print(key, compute_error(wt_compare_dict_1[key]['float'], wt_compare_dict_1[key]['quantized'].dequantize()))
    for key in wt_compare_dict_2:
        print(key, compute_error(wt_compare_dict_2[key]['float'], wt_compare_dict_2[key]['quantized'].dequantize()))

    # Get the backend and save the torchscript files to the specific path
    backend = args.backends
    torch.jit.save(torch.jit.script(net.part1), '{}/part1_{}.torchscript'.format(args.save_dir, backend))
    torch.jit.save(torch.jit.script(net.part2), '{}/part2_{}.torchscript'.format(args.save_dir, backend))
