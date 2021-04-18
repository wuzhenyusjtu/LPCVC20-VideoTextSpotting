# Code for l1 0.5
import sys
import os
import cv2
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.rejector_fots import FOTS_r


def dsample_image(img, ksize):
    h, w = img.shape[:2]
    resized_img = np.lib.stride_tricks.as_strided(
        img,
        shape=(int(h / ksize), int(w / ksize), ksize, ksize, 3),
        strides=img.itemsize * np.array([ksize * w * 3, ksize * 3, w * 3, 1 * 3, 1]))
    return resized_img[:, :, 0, 0].copy()


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


if __name__ == '__main__':
    ''' 
        # Parse the arguments
        # Some useful parameters:
        # trainRoot: Path to training images, which are generated using FOTS model. Image with bounding boxes detected 
        # are marked as 1, otherwise marked as 0.
        # testRoot: Path to testing images.
        # pretrained: Path to pretrained quantized part one for FOTS model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainRoot', required=True, help='path to dataset for training')
    parser.add_argument('--testRoot', required=True, help='path to dataset for testing')
    parser.add_argument('--pretrained', required=True, help="path to pretrained part1 model")
    parser.add_argument('--expr_train_dir', required=True, help='Where to store samples and models')
    parser.add_argument('--expr_test_dir', required=True, help='Where to store samples and models')

    opt = parser.parse_args()
    train_image_dir = opt.trainRoot
    test_image_dir = opt.testRoot
    Part1_pth = opt.pretrained

    feature_model = FOTS_r(part1_path=Part1_pth)

    if os.path.exists(train_image_dir):
        for image_name in os.listdir(train_image_dir):
            print(image_name)
            image = cv2.imread(os.path.join(train_image_dir, image_name))
            image_tensor, scale = pre_process(image)
            _, _, features = feature_model(image_tensor)
            features = F.max_pool2d(features, 2)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            # features = ChannelPool(features)
            features = features[0].flatten().numpy()
            # print(features.shape)
            np.save(opt.expr_train_dir + image_name[:-4] + '.npy', features)

    if os.path.exists(test_image_dir):
        for image_name in os.listdir(test_image_dir):
            print(image_name)
            image = cv2.imread(os.path.join(test_image_dir, image_name))
            image_tensor, scale = pre_process(image)
            _, _, features = feature_model(image_tensor)
            features = F.max_pool2d(features, 2)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features[0].flatten().numpy()
            np.save(opt.expr_test_dir + image_name[:-4] + '.npy', features)
