import copy
import logging
import os.path as osp

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError

from .augmentation import InstanceAugInput, RandomCropWithInstance
from .detection_utils import (annotations_to_instances,
                              build_augmentation,
                              transform_instance_annotations)

"""
This file contains the default mapping that's applied to "dataset dicts".
"""


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for instance in instances:
        bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size

def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    instance = np.random.choice(instances),
    instance = instance[0]
    crop_size = np.asarray(crop_size, dtype=np.int32)
    bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                logger.info("Cannot finished cropping adjustment within 100 tries (#instances {}).".format(len(instances)))
                return T.CropTransform(0, 0, image_size[1], image_size[0])

    return T.CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)

class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        # sem_seg_gt = None

        # aug_input = InstanceAugInput(image, sem_seg=sem_seg_gt, instances=dataset_dict["annotations"])
        # transforms = aug_input.apply_augmentations(self.augmentation)
        # image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        if "annotations" not in dataset_dict or len(dataset_dict["annotations"]) == 0:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            try:
                image, transforms = T.apply_transform_gens([], image)
            except ValueError as e:
                print(dataset_dict["file_name"])
                raise e
            transforms = transforms

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if "annotations" in dataset_dict:

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
