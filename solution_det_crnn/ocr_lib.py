import os
import json
from libs import utils
from libs import dataset
import models.crnn as crnn
import math
import torch
from torch.autograd import Variable
from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image

torch.backends.quantized.engine = 'qnnpack'

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def poly_crop(image, bbox):
    roi_corners = bbox
    x1, x2, y1, y2 = roi_corners[:, 0].min(), roi_corners[:, 0].max(), roi_corners[:, 1].min(), roi_corners[:, 1].max()

    return image[y1: y2, x1: x2, :]


def loadImage(img):  # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


class OCRLib:
    def __init__(self, config_file, det_model_path=None, ocr_model_path=None):
        with open(config_file, "r") as h_config:
            self.config = json.load(h_config)

        dirname = os.path.dirname(config_file)

        if det_model_path is None:
            det_model_path = self.config["det_model"]
            det_model_path = os.path.join(dirname, det_model_path)
        if ocr_model_path is None:
            ocr_model_path = self.config["ocr_model"]
            ocr_model_path = os.path.join(dirname, ocr_model_path)

        self.detector = OCRLib.Detection(
            model_path=det_model_path,
            detection_size=self.config["detection_size"],
            rotated_box=self.config["rotated_box"],
        )

        self.recognizer = OCRLib.Recognition(
            model_path=ocr_model_path,
        )

    def process_rgb_image(self, input_image):
        crop_images = self.detector.get_cropped_images(input_image)
        results = self.recognizer.recognize(crop_images)
        return results

    class Detection:
        def __init__(self, model_path, detection_size, rotated_box):
            self.model = torch.jit.load(model_path)
            self.detection_size = detection_size
            self.rotated_box = rotated_box

        def convert_horizontal_boxes(self, boxes, scale):
            ret_data = []
            boxes = boxes / scale
            for [x1, y1, x2, y2] in boxes.tolist():
                ret_data.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            return ret_data

        def convert_rotated_boxes(self, boxes, scale):
            ret_data = []
            boxes[:, 0:4] = boxes[:, 0:4] / scale
            x_vecs = np.array([-1, 1, 1, -1]) * 0.5
            y_vecs = np.array([-1, -1, 1, 1]) * 0.5
            w_padding = 10
            h_padding = 10
            for cnt_x, cnt_y, w, h, angle in boxes.tolist():
                theta = angle * math.pi / 180.0
                c, s = math.cos(theta), math.sin(theta)
                w = w + w_padding
                h = h + h_padding
                # Rotate boxes
                box_x = cnt_x + (s * y_vecs * h + c * x_vecs * w)
                box_y = cnt_y + (c * y_vecs * h - s * x_vecs * w)
                ret_data.append(np.stack([box_x, box_y], axis=-1))
            return ret_data

        def prepare_inputs(self, input_image):
            scale_ratio = float(self.detection_size) / min(input_image.shape[0:2])
            height = int(input_image.shape[0] * scale_ratio)
            width = int(input_image.shape[1] * scale_ratio)
            im = cv2.resize(input_image, (width, height))

            return (
                torch.from_numpy(im).float().permute(2, 0, 1).unsqueeze(0),
                torch.tensor([[height, width, scale_ratio]]),
            )

        def detect(self, input_image):
            inputs = self.prepare_inputs(input_image)
            boxes, scores, _ = self.model(inputs)

            scale = inputs[1][0][2]
            if self.rotated_box:
                boxes = self.convert_rotated_boxes(boxes, scale)
            else:
                boxes = self.convert_horizontal_boxes(boxes, scale)

            return boxes

        def get_cropped_images(self, image):
            cropped_images = []

            image = loadImage(image)
            height, width = image.shape[:2]
            boxes = self.detect(image)
            for i, box in enumerate(boxes):
                # ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = box
                x1, x2, x3, x4 = np.clip(box[:, 0], 0, width-1)
                y1, y2, y3, y4 = np.clip(box[:, 1], 0, height-1)
                box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                poly = np.array(box).astype(np.int32).reshape((-1))
                poly = poly.reshape(-1, 2)
                img_cropped = poly_crop(image, poly)
                cropped_images.append(img_cropped)
            return cropped_images

    class Recognition:
        def __init__(self, model_path):
            self.model = crnn.CRNN(32, 1, 37, 256)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

        def crnn_inference(self, image):
            alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            converter = utils.strLabelConverter(alphabet)

            transformer = dataset.resizeNormalize((100, 32))
            image = transformer(image)

            image = image.view(1, *image.size())
            image = Variable(image)

            predicts = self.model(image)

            _, predicts = predicts.max(2)
            predicts = predicts.transpose(1, 0).contiguous().view(-1)

            predicts_size = Variable(torch.IntTensor([predicts.size(0)]))
            result = converter.decode(predicts.data, predicts_size.data, raw=False)
            return result

        def recognize(self, crop_images):
            results = []
            for crop in crop_images:
                crop = Image.fromarray(crop).convert('L')
                string = self.crnn_inference(crop)
                results.append(string)
            return results