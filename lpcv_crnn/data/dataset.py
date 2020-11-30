#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import cv2
import os
import random
import json


class SampleDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        
        self.root = root
        self.dir = os.listdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        self.trans_table = dict.fromkeys(map(ord, '!*^&$@\',.:;-'), None)
        
    def __len__(self):
        return len(self.dir)

    def __getitem__(self, index):
        img_name = self.dir[index]
        if not img_name.endswith('.jpg'):
            return self.__getitem__(random.randint(0, len(self.dir)))
        img = cv2.imread(os.path.join(self.root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.transform is not None:
            img = self.transform(img)
        label = str(img_name.split(':')[0]).translate(self.trans_table).lower()
#         label = str(img_name.split(':')[0])
        return (img, label)


class SynthDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        
        self.root = root
        self.dir = os.listdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.dir)

    def __getitem__(self, index):
        img_name = self.dir[index]
        if not img_name.endswith('.jpg'):
            print('{} name error',format(img_name))
            return self.__getitem__(random.randint(0, len(self.dir)))
        try:
            img = cv2.imread(os.path.join(self.root, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            print('{} error',format(img_name))
            return self.__getitem__(random.randint(0, len(self.dir)))
        
#         print('{} been used'.format(img_name))
        if self.transform is not None:
            img = self.transform(img)
        label = str(img_name.split('`')[0])
        return (img, label)


class MJDataset(Dataset):

    def __init__(self, jsonpath=None, transform=None, target_transform=None):
        
        self.jsonpath = jsonpath
        with open(self.jsonpath, 'r') as file:
            self.path_list = json.load(file)
        self.transform = transform
        self.target_transform = target_transform
        
        self.dirname = os.path.dirname(self.jsonpath)
        self.new_path_list = []
        for img_name in self.path_list:
            self.new_path_list.append(os.path.join(self.dirname, img_name))
        
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        img_name = self.new_path_list[index]
        try:
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            print('{} error',format(img_name))
            return self.__getitem__(random.randint(0, len(self.path_list)))
        
#         print('{} been used'.format(img_name))
        if self.transform is not None:
            img = self.transform(img)
        label = os.path.basename(os.path.basename(img_name).split('_')[1])
        return (img, label)  
    
# class MJDataset(Dataset):


class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
#         img = img.resize(self.size, self.interpolation)
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
