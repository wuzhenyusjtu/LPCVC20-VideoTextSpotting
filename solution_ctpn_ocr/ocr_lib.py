import json
from math import *

import numpy as np
import torch
import torch.nn.functional as F
import os
from ctpn import CTPN_Model
from utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox
from utils import resize, nms, TextProposalConnectorOriented

import cv2
import math

torch.backends.quantized.engine = 'qnnpack'


def sort_box(box):
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


def parse_boxes(boxes):
    reformat_boxes = []
    for box in boxes:
        box = box[:8]
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        reformat_boxes.append(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
    return reformat_boxes


class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '_'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


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
            image_height=self.config["image_height"],
            hs_factor=self.config["hs_factor"],
            charset=self.config["charset"],
        )

    def process_rgb_image(self, input_image):
        boxes = self.detector.get_det_boxes(input_image)
        boxes = sort_box(boxes)
        boxes = parse_boxes(boxes)
        results = []
        for box in boxes:
            result = self.recognizer.recognize(input_image, box)
            results.append(result)
        return results

    class Detection:
        def __init__(self, model_path, detection_size, rotated_box, device='cpu'):
            self.model = CTPN_Model()
            self.model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            self.detection_size = detection_size
            self.rotated_box = rotated_box

        def get_det_boxes(self, image, expand=True, height=720, prob_thresh=0.5, device='cpu'):
            image = resize(image, height=height)
            h, w = image.shape[:2]
            image = image.astype(np.float32) - [123.68, 116.779, 103.939]
            image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

            with torch.no_grad():
                image = image.to(device)
                cls, regr = self.model(image)
                cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
                regr = regr.cpu().numpy()
                anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
                bbox = bbox_transfor_inv(anchor, regr)
                bbox = clip_box(bbox, [h, w])
                # print(bbox.shape)

                fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
                # print(np.max(cls_prob[0, :, 1]))
                select_anchor = bbox[fg, :]
                select_score = cls_prob[0, fg, 1]
                select_anchor = select_anchor.astype(np.int32)
                # print(select_anchor.shape)
                keep_index = filter_bbox(select_anchor, 16)

                # nms
                select_anchor = select_anchor[keep_index]
                select_score = select_score[keep_index]
                select_score = np.reshape(select_score, (select_score.shape[0], 1))
                nmsbox = np.hstack((select_anchor, select_score))
                keep = nms(nmsbox, 0.3)
                # print(keep)
                select_anchor = select_anchor[keep]
                select_score = select_score[keep]

                # text line-
                textConn = TextProposalConnectorOriented()
                text = textConn.get_text_lines(select_anchor, select_score, [h, w])

                # expand text
                if expand:
                    for idx in range(len(text)):
                        text[idx][0] = max(text[idx][0] - 10, 0)
                        text[idx][2] = min(text[idx][2] + 10, w - 1)
                        text[idx][4] = max(text[idx][4] - 10, 0)
                        text[idx][6] = min(text[idx][6] + 10, w - 1)

                return text

    class Recognition:
        def __init__(self, model_path, image_height, hs_factor, charset=None):
            self.rcg_module = torch.jit.load(model_path)
            self.image_height = image_height
            self.hs_factor = hs_factor
            self.charset = charset

        def prepare_inputs(self, input_image, box):
            # Find homography transform and apply
            b_w = np.linalg.norm(box[0] - box[1])
            b_h = np.linalg.norm(box[1] - box[2])
            t_h = self.image_height
            t_w = math.floor(b_w * t_h / b_h * self.hs_factor)
            trgt_box = np.array(
                [[0, 0], [t_w, 0], [t_w, t_h], [0, t_h]], dtype="float32"
            )

            # Apply transform
            xfm_mat = cv2.getPerspectiveTransform(
                np.array(box, dtype="float32"), trgt_box
            )
            text_patch = cv2.warpPerspective(input_image, xfm_mat, (t_w, t_h))
            return (
                    torch.from_numpy(text_patch).float().permute(2, 0, 1).unsqueeze(0)
                    / 255.0
            )

        def recognize(self, input_image, box):
            # Setup inputs
            inputs = self.prepare_inputs(input_image, box)
            symbols, score = self.rcg_module(inputs)

            symbols = symbols.tolist()
            # score = score.item()
            decode_str = "".join([self.charset[int(x)] for x in symbols])

            return decode_str
