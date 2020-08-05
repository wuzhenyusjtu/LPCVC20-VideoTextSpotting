import math
import os
import random
import re

import cv2
import numpy as np
import scipy.io
import torch
import torch.utils.data
import torchvision
from shapely.geometry import Polygon, box
import shapely


def point_dist_to_line(p1, p2, p3):
    """Compute the distance from p3 to p2-p1."""
    if not np.array_equal(p1, p2):
        return np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    else:
        return np.linalg.norm(p3 - p1)


IN_OUT_RATIO = 4
IN_SIDE = 280
OUT_SIDE = int(IN_SIDE // IN_OUT_RATIO)

def transform(im, quads, texts, normalizer, data_set):
    scale = 320 / np.minimum(im.shape[0], im.shape[1])
    upscaled = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    quads = quads * scale
    # rotate
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = upscaled.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    angle = torch.empty(1).uniform_(-10, 10).item()
    M = cv2.getRotationMatrix2D((cX, cY), angle=angle, scale=1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))  # TODO replace with round and do it later
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(upscaled, M, (nW, nH))
    quads = cv2.transform(quads, M)
    # stretch
    strechK = torch.empty(1).uniform_(0.9, 1.2).item()
    stretched = cv2.resize(rotated, None, fx=1, fy=strechK, interpolation=cv2.INTER_CUBIC)
    quads[:, :, 1] = quads[:, :, 1] * strechK
    
    quads /= IN_OUT_RATIO

    training_mask = np.ones((OUT_SIDE, OUT_SIDE), dtype=float)
    classification = np.zeros((OUT_SIDE, OUT_SIDE), dtype=float)
    regression = np.zeros((4,) + classification.shape, dtype=float)
    tmp_cls = np.empty(classification.shape, dtype=float)
    thetas = np.zeros(classification.shape, dtype=float)

    # crop
    crop_max_y = stretched.shape[0] // IN_OUT_RATIO - OUT_SIDE  # since Synth has some low images, there is a chance that y coord of crop can be zero only
    if 0 != crop_max_y:
        crop_point = (torch.randint(low=0, high=stretched.shape[1] // IN_OUT_RATIO - OUT_SIDE, size=(1,), dtype=torch.int16).item(),
                      torch.randint(low=0, high=stretched.shape[0] // IN_OUT_RATIO - OUT_SIDE, size=(1,), dtype=torch.int16).item())
    else:
        crop_point = (torch.randint(low=0, high=stretched.shape[1] // IN_OUT_RATIO - OUT_SIDE, size=(1,), dtype=torch.int16).item(),
                      0)
    crop_box = box(crop_point[0], crop_point[1], crop_point[0] + OUT_SIDE, crop_point[1] + OUT_SIDE)
    for quad_id, quad in enumerate(quads):
        polygon = Polygon(quad)
        try:
            intersected_polygon = polygon.intersection(crop_box)
        except shapely.errors.TopologicalError:  # some points of quads in Synth can be in wrong order
            quad[1], quad[2] = quad[2], quad[1]
            polygon = Polygon(quad)
            intersected_polygon = polygon.intersection(crop_box)
        if intersected_polygon.is_empty:
            continue
        if type(intersected_polygon) is Polygon:
            intersected_quad = np.array(intersected_polygon.exterior.coords[:-1])
            intersected_quad -= crop_point
            intersected_minAreaRect = cv2.minAreaRect(intersected_quad.astype(np.float32))
            intersected_minAreaRect_boxPoints = cv2.boxPoints(intersected_minAreaRect)
            cv2.fillConvexPoly(training_mask, intersected_minAreaRect_boxPoints.round().astype(int), 0)
            minAreaRect = cv2.minAreaRect(quad.astype(np.float32))
            shrinkage = min(minAreaRect[1][0], minAreaRect[1][1]) * 0.6
            shrunk_width_and_height = (intersected_minAreaRect[1][0] - shrinkage, intersected_minAreaRect[1][1] - shrinkage)
            if shrunk_width_and_height[0] >= 0 and shrunk_width_and_height[1] >= 0 and texts[quad_id]:
                shrunk_minAreaRect = intersected_minAreaRect[0], shrunk_width_and_height, intersected_minAreaRect[2]

                poly = intersected_minAreaRect_boxPoints
                if intersected_minAreaRect[2] >= -45:
                    poly = np.array([poly[1], poly[2], poly[3], poly[0]])
                else:
                    poly = np.array([poly[2], poly[3], poly[0], poly[1]])
                angle_cos = (poly[2, 0] - poly[3, 0]) / np.sqrt(
                    (poly[2, 0] - poly[3, 0]) ** 2 + (poly[2, 1] - poly[3, 1]) ** 2 + 1e-5)  # TODO tg or ctg
                angle = np.arccos(angle_cos)
                if poly[2, 1] > poly[3, 1]:
                    angle *= -1
                angle += 45 * np.pi / 180

                tmp_cls.fill(0)
                round_shrink_minAreaRect_boxPoints = cv2.boxPoints(shrunk_minAreaRect)
                cv2.fillConvexPoly(tmp_cls, round_shrink_minAreaRect_boxPoints.round(out=round_shrink_minAreaRect_boxPoints).astype(int), 1)
                cv2.rectangle(tmp_cls, (0, 0), (tmp_cls.shape[1] - 1, tmp_cls.shape[0] - 1), 0, thickness=int(round(shrinkage * 2)))

                classification += tmp_cls
                training_mask += tmp_cls
                thetas += tmp_cls * angle

                points = np.nonzero(tmp_cls)
                pointsT = np.transpose(points)
                for point in pointsT:
                    for plane in range(3):  # TODO widht - dist, height - other dist and more percise dist
                        regression[(plane,) + tuple(point)] = point_dist_to_line(poly[plane], poly[plane + 1], np.array((point[1], point[0]))) * IN_OUT_RATIO
                    regression[(3,) + tuple(point)] = point_dist_to_line(poly[3], poly[0], np.array((point[1], point[0]))) * IN_OUT_RATIO
    if 0 == np.count_nonzero(classification) and 0.1 < torch.rand(1).item():
        return data_set[torch.randint(low=0, high=len(data_set), size=(1,), dtype=torch.int16).item()]
    # avoiding training on black corners decreases hmean, see d9c727a8defbd1c8022478ae798c907ccd2fa0b2. This may happen
    # because of OHEM: it already guides the training and it won't select back corner pixels if the net is good at
    # classifying them. It can be easily verified by removing OHEM, but I didn't test it
    cropped = stretched[crop_point[1] * IN_OUT_RATIO:crop_point[1] * IN_OUT_RATIO + IN_SIDE, crop_point[0] * IN_OUT_RATIO:crop_point[0] * IN_OUT_RATIO + IN_SIDE]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    permuted = np.transpose(cropped, (2, 0, 1))
    permuted = torch.from_numpy(permuted).float()
    permuted = normalizer(permuted)
    return permuted, torch.from_numpy(classification).float(), torch.from_numpy(regression).float(), torch.from_numpy(
        thetas).float(), torch.from_numpy(training_mask).float()

IN_OUT_RATIO_ft = 4
IN_SIDE_ft = 300
OUT_SIDE_ft = IN_SIDE_ft // IN_OUT_RATIO_ft

def transform_finetune(im, quads, texts, normalizer):
    # upscale
    scale = 280 / np.minimum(im.shape[0], im.shape[1])
    upscaled = cv2.resize(im, None, fx=scale, fy=scale , interpolation=cv2.INTER_CUBIC)
    quads = quads * scale
    
    IN_SIDE_X = upscaled.shape[0]
    IN_SIDE_Y = upscaled.shape[1]
    OUT_SIDE_X = IN_SIDE_X // IN_OUT_RATIO_ft
    OUT_SIDE_Y = IN_SIDE_Y // IN_OUT_RATIO_ft
    # rotate
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = upscaled.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    # stretch
    stretched = upscaled
    
    quads /= IN_OUT_RATIO_ft
    
    training_mask = np.ones((OUT_SIDE_X, OUT_SIDE_Y), dtype=float)
    classification = np.zeros((OUT_SIDE_X, OUT_SIDE_Y), dtype=float)
    regression = np.zeros((4,) + classification.shape, dtype=float)
    tmp_cls = np.empty(classification.shape, dtype=float)
    thetas = np.zeros(classification.shape, dtype=float)
    
    crop_point = (0, 0)
    crop_box = box(0, 0, OUT_SIDE_Y, OUT_SIDE_X)
    for quad_id, quad in enumerate(quads):
        polygon = Polygon(quad)
        try:
            intersected_polygon = polygon.intersection(crop_box)
        except shapely.errors.TopologicalError:
            quad[1], quad[2] = quad[2], quad[1]
            polygon = Polygon(quad)
            intersected_polygon = polygon.intersection(crop_box)
        if intersected_polygon.is_empty:
            print('{} quads empty'.format(quad))
            continue
        if type(intersected_polygon) is Polygon:
            intersected_quad = np.array(intersected_polygon.exterior.coords[:-1])
            intersected_quad -= crop_point
            intersected_minAreaRect = cv2.minAreaRect(intersected_quad.astype(np.float32))
            intersected_minAreaRect_boxPoints = cv2.boxPoints(intersected_minAreaRect)
            cv2.fillConvexPoly(training_mask, intersected_minAreaRect_boxPoints.round().astype(int), 0)
            minAreaRect = cv2.minAreaRect(quad.astype(np.float32))
            shrinkage = min(minAreaRect[1][0], minAreaRect[1][1]) * 0.6
            shrunk_width_and_height = (intersected_minAreaRect[1][0] - shrinkage, intersected_minAreaRect[1][1] - shrinkage)
            if shrunk_width_and_height[0] >= 0 and shrunk_width_and_height[1] >= 0 and texts[quad_id]:
                shrunk_minAreaRect = intersected_minAreaRect[0], shrunk_width_and_height, intersected_minAreaRect[2]

                poly = intersected_minAreaRect_boxPoints
                if intersected_minAreaRect[2] >= -45:
                    poly = np.array([poly[1], poly[2], poly[3], poly[0]])
                else:
                    poly = np.array([poly[2], poly[3], poly[0], poly[1]])
                angle_cos = (poly[2, 0] - poly[3, 0]) / np.sqrt(
                    (poly[2, 0] - poly[3, 0]) ** 2 + (poly[2, 1] - poly[3, 1]) ** 2 + 1e-5)  # TODO tg or ctg
                angle = np.arccos(angle_cos)
                if poly[2, 1] > poly[3, 1]:
                    angle *= -1
                angle += 45 * np.pi / 180

                tmp_cls.fill(0)
                round_shrink_minAreaRect_boxPoints = cv2.boxPoints(shrunk_minAreaRect)
                cv2.fillConvexPoly(tmp_cls, round_shrink_minAreaRect_boxPoints.round(out=round_shrink_minAreaRect_boxPoints).astype(int), 1)
                cv2.rectangle(tmp_cls, (0, 0), (tmp_cls.shape[1] - 1, tmp_cls.shape[0] - 1), 0, thickness=int(round(shrinkage * 2)))

                classification += tmp_cls
                training_mask += tmp_cls
                thetas += tmp_cls * angle

                points = np.nonzero(tmp_cls)
                pointsT = np.transpose(points)
                for point in pointsT:
                    for plane in range(3):  # TODO widht - dist, height - other dist and more percise dist
                        regression[(plane,) + tuple(point)] = point_dist_to_line(poly[plane], poly[plane + 1], np.array((point[1], point[0]))) * IN_OUT_RATIO_ft
                    regression[(3,) + tuple(point)] = point_dist_to_line(poly[3], poly[0], np.array((point[1], point[0]))) * IN_OUT_RATIO_ft
    # avoiding training on black corners decreases hmean, see d9c727a8defbd1c8022478ae798c907ccd2fa0b2. This may happen
    # because of OHEM: it already guides the training and it won't select back corner pixels if the net is good at
    # classifying them. It can be easily verified by removing OHEM, but I didn't test it
    cropped = stretched[crop_point[1] * IN_OUT_RATIO_ft:crop_point[1] * IN_OUT_RATIO_ft + IN_SIDE_X, crop_point[0] * IN_OUT_RATIO_ft:crop_point[0] * IN_OUT_RATIO_ft + IN_SIDE_Y]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    permuted = np.transpose(cropped, (2, 0, 1))
    permuted = torch.from_numpy(permuted).float()
    permuted = normalizer(permuted)
    return permuted, torch.from_numpy(classification).float(), torch.from_numpy(regression).float(), torch.from_numpy(
        thetas).float(), torch.from_numpy(training_mask).float()

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = root
        self.img_dir = 'train_images1'
        self.labels_dir = 'sample_train_annotation_txt'
        self.image_prefix = []
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
        for dirEntry in os.scandir(os.path.join(root, self.img_dir)):
            self.image_prefix.append(dirEntry.name[:-4])
        self.transform = transform
    def __len__(self):
        return len(self.image_prefix)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(os.path.join(self.root, self.img_dir), self.image_prefix[idx] + '.jpg'), cv2.IMREAD_COLOR).astype(np.float32)
        quads = []
        texts = []
        lines = [line.rstrip('\n') for line in open(os.path.join(os.path.join(self.root,self.labels_dir), self.image_prefix[idx] + '.txt'),
                                                    encoding='utf-8-sig')]
        for line in lines:
            matches = line.split(',')
            numbers = np.array(matches[:8], dtype=float)
            quads.append(numbers.reshape((4, 2)))
            texts.append('###' != matches[8])
        if len(quads) == 0:
            return self[torch.randint(low=0, high=len(self), size=(1,), dtype=torch.int16).item()]
        return self.transform(img, np.stack(quads), texts, self.normalizer)

class SampleDatasetVal(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = root
        self.img_dir = 'test_images1'
        self.labels_dir = 'sample_test_annotation_txt'
        self.image_prefix = []
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
        for dirEntry in os.scandir(os.path.join(root, self.img_dir)):
            self.image_prefix.append(dirEntry.name[:-4])
        self.transform = transform
    def __len__(self):
        return len(self.image_prefix)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(os.path.join(self.root, self.img_dir), self.image_prefix[idx] + '.jpg'), cv2.IMREAD_COLOR).astype(np.float32)
        quads = []
        texts = []
        lines = [line.rstrip('\n') for line in open(os.path.join(os.path.join(self.root,self.labels_dir), self.image_prefix[idx] + '.txt'),
                                                    encoding='utf-8-sig')]
        for line in lines:
            matches = line.split(',')
            numbers = np.array(matches[:8], dtype=float)
            quads.append(numbers.reshape((4, 2)))
            texts.append('###' != matches[8])
        if len(quads) == 0:
            return self[torch.randint(low=0, high=len(self), size=(1,), dtype=torch.int16).item()]
        return self.transform(img, np.stack(quads), texts, self.normalizer)

class MergeText(torch.utils.data.Dataset):
    def __init__(self, root_syn, root_sample, syn_transform, sample_transform):
        
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
        
        # SynthText dataset code
        self.syn_transform = syn_transform
        self.syn_root = root_syn
        self.syn_labels = scipy.io.loadmat(os.path.join(root_syn, 'gt.mat'))
        self.syn_broken_image_ids = set()
        self.syn_len = self.syn_labels['imnames'].shape[1] // 60
        self.syn_len = int(0.8*self.syn_len)
        
        # Sample dataset code
        self.sample_transform = sample_transform
        self.sample_root = root_sample
        self.sample_img_dir = 'train_images1'
        self.sample_labels_dir = 'sample_train_annotation_txt'
        self.sample_image_prefix = []

        for dirEntry in os.scandir(os.path.join(self.sample_root, self.sample_img_dir)):
            self.sample_image_prefix.append(dirEntry.name[:-4])
        self.sample_len = len(self.sample_image_prefix)
        
        
    def __len__(self):
        return self.syn_len + self.sample_len
    
    def __getitem__(self, idx):
        # Get SynthText image
        if idx < self.syn_len:
            idx = (idx * 60) + random.randint(0, 59)
            if idx in self.syn_broken_image_ids:
                return self[torch.randint(low=0, high=self.syn_len, size=(1,), dtype=torch.int16).item()]
            img = cv2.imread(os.path.join(self.syn_root, self.syn_labels['imnames'][0, idx][0]), cv2.IMREAD_COLOR).astype(np.float32)
            if 190 >= img.shape[0]:  # image is too low, it will not be possible to crop 640x640 after transformations
                self.syn_broken_image_ids.add(idx)
                return self[torch.randint(low=0, high=self.syn_len, size=(1,), dtype=torch.int16).item()]
            coordinates = self.syn_labels['wordBB'][0, idx]
            if len(coordinates.shape) == 2:
                coordinates = np.expand_dims(coordinates, axis=2)
            transposed = np.transpose(coordinates, (2, 1, 0))
            if (transposed > 0).all() and (transposed[:, :, 1] < img.shape[1]).all() and (transposed[:, :, 1] < img.shape[0]).all():
                if ((transposed[:, 0] != transposed[:, 1]).all() and
                    (transposed[:, 0] != transposed[:, 2]).all() and
                    (transposed[:, 0] != transposed[:, 3]).all() and
                    (transposed[:, 1] != transposed[:, 2]).all() and
                    (transposed[:, 1] != transposed[:, 3]).all() and
                    (transposed[:, 2] != transposed[:, 3]).all()):  # boxes can be in a form [p1, p1, p2, p2], while we need [p1, p2, p3, p4]
                        return self.syn_transform(img, transposed, (True, ) * len(transposed), self.normalizer, self)
            self.syn_broken_image_ids.add(idx)
            return self[torch.randint(low=0, high=self.syn_len, size=(1,), dtype=torch.int16).item()]
        else:
            img = cv2.imread(os.path.join(os.path.join(self.sample_root, self.sample_img_dir), self.sample_image_prefix[idx-self.syn_len] + '.jpg'), cv2.IMREAD_COLOR).astype(np.float32)
            quads = []
            texts = []
            lines = [line.rstrip('\n') for line in open(os.path.join(os.path.join(self.sample_root,self.sample_labels_dir), self.sample_image_prefix[idx-self.syn_len] + '.txt'),encoding='utf-8-sig')]
            for line in lines:
                matches = line.split(',')
                numbers = np.array(matches[:8], dtype=float)
                quads.append(numbers.reshape((4, 2)))
                texts.append('###' != matches[8])
            if len(quads) == 0:
                return self[torch.randint(low=self.syn_len, high=len(self), size=(1,), dtype=torch.int16).item()]
            return self.sample_transform(img, np.stack(quads), texts, self.normalizer, self)

class MergeTextVal(torch.utils.data.Dataset):
    def __init__(self, root_syn, root_sample, syn_transform, sample_transform):
        
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
        
        # SynthText dataset code
        self.syn_transform = syn_transform
        self.syn_root = root_syn
        self.syn_labels = scipy.io.loadmat(os.path.join(root_syn, 'gt.mat'))
        self.syn_broken_image_ids = set()
        self.syn_len = self.syn_labels['imnames'].shape[1] // 60
        self.syn_len = int(0.2*self.syn_len)
        self.syn_train_len = int(0.8*self.syn_len)
        
        # Sample dataset code
        self.sample_transform = sample_transform
        self.sample_root = root_sample
        self.sample_img_dir = 'test_images1'
        self.sample_labels_dir = 'sample_test_annotation_txt'
        self.sample_image_prefix = []

        for dirEntry in os.scandir(os.path.join(self.sample_root, self.sample_img_dir)):
            self.sample_image_prefix.append(dirEntry.name[:-4])
        self.sample_len = len(self.sample_image_prefix)
        
        
    def __len__(self):
        return self.syn_len + self.sample_len
    
    def __getitem__(self, idx):
        # Get SynthText image
        if idx < self.syn_len:
            idx = (idx * 60) + random.randint(0, 59) + self.syn_train_len
            if idx in self.syn_broken_image_ids:
                return self[torch.randint(low=0, high=self.syn_len, size=(1,), dtype=torch.int16).item()]
            img = cv2.imread(os.path.join(self.syn_root, self.syn_labels['imnames'][0, idx][0]), cv2.IMREAD_COLOR).astype(np.float32)
            if 190 >= img.shape[0]:  # image is too low, it will not be possible to crop 640x640 after transformations
                self.syn_broken_image_ids.add(idx)
                return self[torch.randint(low=0, high=self.syn_len, size=(1,), dtype=torch.int16).item()]
            coordinates = self.syn_labels['wordBB'][0, idx]
            if len(coordinates.shape) == 2:
                coordinates = np.expand_dims(coordinates, axis=2)
            transposed = np.transpose(coordinates, (2, 1, 0))
            if (transposed > 0).all() and (transposed[:, :, 1] < img.shape[1]).all() and (transposed[:, :, 1] < img.shape[0]).all():
                if ((transposed[:, 0] != transposed[:, 1]).all() and
                    (transposed[:, 0] != transposed[:, 2]).all() and
                    (transposed[:, 0] != transposed[:, 3]).all() and
                    (transposed[:, 1] != transposed[:, 2]).all() and
                    (transposed[:, 1] != transposed[:, 3]).all() and
                    (transposed[:, 2] != transposed[:, 3]).all()):  # boxes can be in a form [p1, p1, p2, p2], while we need [p1, p2, p3, p4]
                        return self.syn_transform(img, transposed, (True, ) * len(transposed), self.normalizer)
            self.syn_broken_image_ids.add(idx)
            return self[torch.randint(low=0, high=self.syn_len, size=(1,), dtype=torch.int16).item()]
        else:
            img = cv2.imread(os.path.join(os.path.join(self.sample_root, self.sample_img_dir), self.sample_image_prefix[idx-self.syn_len] + '.jpg'), cv2.IMREAD_COLOR).astype(np.float32)
            quads = []
            texts = []
            lines = [line.rstrip('\n') for line in open(os.path.join(os.path.join(self.sample_root,self.sample_labels_dir), self.sample_image_prefix[idx-self.syn_len] + '.txt'),encoding='utf-8-sig')]
            for line in lines:
                matches = line.split(',')
                numbers = np.array(matches[:8], dtype=float)
                quads.append(numbers.reshape((4, 2)))
                texts.append('###' != matches[8])
            if len(quads) == 0:
                return self[torch.randint(low=self.syn_len, high=len(self), size=(1,), dtype=torch.int16).item()]
            return self.sample_transform(img, np.stack(quads), texts, self.normalizer)
        
class SynthText(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = root
        self.labels = scipy.io.loadmat(os.path.join(root, 'gt.mat'))
        self.broken_image_ids = set()
        self.pattern = re.compile('^' + '(\\d+),' * 8 + '(.+)$')
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.labels['imnames'].shape[1] // 105  # there are more than 105 text images for each source image

    def __getitem__(self, idx):
        idx = (idx * 105) + random.randint(0, 104)  # compensate dataset size, while maintain diversity
        if idx in self.broken_image_ids:
            return self[torch.randint(low=0, high=len(self), size=(1,), dtype=torch.int16).item()]
        img = cv2.imread(os.path.join(self.root, self.labels['imnames'][0, idx][0]), cv2.IMREAD_COLOR).astype(np.float32)
        if 190 >= img.shape[0]:  # image is too low, it will not be possible to crop 640x640 after transformations
            self.broken_image_ids.add(idx)
            return self[torch.randint(low=0, high=len(self), size=(1,), dtype=torch.int16).item()]
        coordinates = self.labels['wordBB'][0, idx]
        if len(coordinates.shape) == 2:
            coordinates = np.expand_dims(coordinates, axis=2)
        transposed = np.transpose(coordinates, (2, 1, 0))
        if (transposed > 0).all() and (transposed[:, :, 1] < img.shape[1]).all() and (transposed[:, :, 1] < img.shape[0]).all():
            if ((transposed[:, 0] != transposed[:, 1]).all() and
                (transposed[:, 0] != transposed[:, 2]).all() and
                (transposed[:, 0] != transposed[:, 3]).all() and
                (transposed[:, 1] != transposed[:, 2]).all() and
                (transposed[:, 1] != transposed[:, 3]).all() and
                (transposed[:, 2] != transposed[:, 3]).all()):  # boxes can be in a form [p1, p1, p2, p2], while we need [p1, p2, p3, p4]
                    return transform(img, transposed, (True, ) * len(transposed), self.normalizer)
        self.broken_image_ids.add(idx)
        return self[torch.randint(low=0, high=len(self), size=(1,), dtype=torch.int16).item()]


if '__main__' == __name__:
    print("test")
