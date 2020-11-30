# Code for l1 0.5 
import sys 
sys.path.append("..") 

import argparse

import cv2
import numpy as np
import os
import torch
import torch.utils.data
import tqdm

from models.quanted_fots import FOTS_quanted

import sys 
sys.path.append("..") 



def evaluate_net(model, images_folder):
    model.eval()
    cnt = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(os.listdir(images_folder)[0:5], desc='Test', ncols=80)
        for image_name in pbar:
            if not image_name.startswith('res'):
                print(image_name)
                continue
            prefix = image_name[:image_name.rfind('.')]
            image = cv2.imread(os.path.join(images_folder, image_name), cv2.IMREAD_COLOR)
            scale_y = 270 / image.shape[0]  # 1248 # 704
            scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale_y, fy=scale_y, interpolation=cv2.INTER_CUBIC)
            orig_scaled_image = scaled_image.copy()

            scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
            scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            image_tensor = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()

            confidence, distances, angle = net(image_tensor)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate-folder', type=str, required=True, help='Path to folder with images for calibrating')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--pretrain-model', type=str, default=None, help='Pretrained model name in save dir')
    parser.add_argument('--backends', type=str, default='fbgemm', help='Quantization backends')
    args = parser.parse_args()

    net = FOTS_quanted()
    
    checkpoint_name = args.pretrain_model
    checkpoint = torch.load(checkpoint_name, map_location='cpu')
    
    net.load_state_dict(checkpoint, strict=False)
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    
    # Must set eval
    net.eval()
    
    # Get quantization configure
    qcf = torch.quantization.get_default_qconfig(args.backends)
    
    # Set quantization configure to all parts in FOTs model
    net.rejector.qconfig = qcf
    net.part1.qconfig = qcf
    net.part2.qconfig = qcf
    net.part3.qconfig = qcf
    
    # Fuse all parts in the FOTs model
    net.fuse_model()
    
    # Prepare the quantization for all parts
    torch.quantization.prepare(net.rejector, inplace=True)
    torch.quantization.prepare(net.part1, inplace=True)
    torch.quantization.prepare(net.part2, inplace=True)
    torch.quantization.prepare(net.part3, inplace=True)
    
    # Use some images from the folder for evaluation or calibration
    images_folder = args.calibrate_folder
    evaluate_net(net, images_folder)
    
    # Convert all parts into quantized ones
    torch.quantization.convert(net.rejector, inplace=True)
    torch.quantization.convert(net.part1, inplace=True)
    torch.quantization.convert(net.part2, inplace=True)
    torch.quantization.convert(net.part3, inplace=True)
    
    # Get the backend and save the torchscript files to the specific path
    backend = args.backends
    torch.jit.save(torch.jit.script(net.rejector), '{}/rejector_{}.torchscript'.format(args.save_dir, backend))
    torch.jit.save(torch.jit.script(net.part1), '{}/part1_{}.torchscript'.format(args.save_dir, backend))
    torch.jit.save(torch.jit.script(net.part2), '{}/part2_{}.torchscript'.format(args.save_dir, backend))
    torch.jit.save(torch.jit.script(net.part3), '{}/part3_{}.torchscript'.format(args.save_dir, backend))
