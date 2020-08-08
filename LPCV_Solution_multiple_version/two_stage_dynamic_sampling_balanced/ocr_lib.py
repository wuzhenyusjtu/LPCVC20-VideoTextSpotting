#!python3

import json
import math

import cv2
import numpy as np
import torch
import os
from modules.parse_polys import parse_polys

import crnn
import utils
from torch.autograd import Variable
import torch.nn as nn
from scipy.signal import find_peaks
from fots import FOTSModel_q
from model_rej.model_inf import ABCNet_inf
from model_rej.configs import get_cfg
import pickle

torch.backends.quantized.engine = 'fbgemm'
SAMPLE_RATE_DICT = {'NT': 90, 'LQ': 60, 'HQ': 5}
cache_num = 2


class OCRLib:
    def __init__(self, config_file, det_model_path=None, ocr_model_path=None):
        with open(config_file, "r") as h_config:
            self.config = json.load(h_config)

        dirname = os.path.dirname(config_file)

        if det_model_path is None:
            det_model_part1_path = os.path.join(dirname, self.config["det_model_part1"])
            det_model_part2_path = os.path.join(dirname, self.config["det_model_part2"])
            det_model_part3_path = os.path.join(dirname, self.config["det_model_part3"])
            det_model_rejector_path = os.path.join(dirname, self.config["det_model_rejector"])
            det_model_fc_path = os.path.join(dirname, self.config["det_model_fc"])
            thrshold = self.config["rejector_thrs"]

        if ocr_model_path is None:
            ocr_model_cnn_path = os.path.join(dirname, self.config['ocr_cnn_model'])
            ocr_model_ew0_path = os.path.join(dirname, self.config['ocr_ew0_model'])
            ocr_model_ew1_path = os.path.join(dirname, self.config['ocr_ew1_model'])
            ocr_model_path = os.path.join(dirname, self.config['ocr_model'])

        self.selector = OCRLib.Selection(
            svm_model_path='models/svm_layer_1.p',
            abcnet_model_path='model_rej/model_mobile_all_bn_new.pth',
        )

        self.detector = OCRLib.Detection(
            part1_path=det_model_part1_path,
            part2_path=det_model_part2_path,
            part3_path=det_model_part3_path,
            rejector_path=det_model_rejector_path,
            fc_path=det_model_fc_path,
            thrshold=thrshold
        )

        self.recognizer = OCRLib.Recognition(
            model_path=ocr_model_path,
            model_cnn_path=ocr_model_cnn_path,
            model_ew0_path=ocr_model_ew0_path,
            model_ew1_path=ocr_model_ew1_path,
            charset=self.config["charset"],
        )

        self.cnt = 0

    def process_rgb_image(self, input_image, cache_boxes_list, cache_images, state_flag):
        input_image = self.selector.select(input_image)
        results = []
        if input_image is not None:
            boxes = self.detector.detect(input_image)
            if len(boxes) > 0:
                # Build a cache to store boxes
                if state_flag == 'HQ':
                    # Set the number of cache_images to use
                    if len(cache_boxes_list) < cache_num:
                        if len(min(cache_boxes_list, key=len)) > len(boxes):
                            # The drone is leaving the scene while caching did not reach caching number
                            max_boxes = max(cache_boxes_list, key=len)
                            max_ind = cache_boxes_list.index(max_boxes)
                            select_images = cache_images[max_ind]
                            for box in max_boxes:
                                result = self.recognizer.recognize(select_images, box)
                                results.append(result)
                            # Quit the scene, set the state_flag to 'NT'
                            state_flag = 'NT'
                            return results, state_flag, cache_boxes_list.clear()
                        cache_boxes_list.append(boxes)
                        # Continue caching frame of the current scene, set the state_flag to 'HQ'
                        state_flag = 'HQ'
                    else:
                        # Find boxes in cache_boxes_list with max len and pass the cache images to recognizer
                        max_boxes = max(cache_boxes_list, key=len)
                        max_ind = cache_boxes_list.index(max_boxes)
                        select_images = cache_images[max_ind]
                        for box in max_boxes:
                            result = self.recognizer.recognize(select_images, box)
                            results.append(result)
                            # Already get clear image of the scene, set the state_flag to 'NT'
                        state_flag = 'NT'
                        cache_boxes_list.clear()
                else:
                    cache_boxes_list.append(boxes)
                    # Detect the first frame of the current scene, set the state_flag to 'HQ'
                    state_flag = 'HQ'
            else:
                # Not rejected by two rejecter but no boxes detected, current frame is heavy blur images
                state_flag = 'LQ'
        else:
            # Rejected by the rejecter, current frame is non_text or heavy blur images
            state_flag = 'NT'
        return results, state_flag, cache_boxes_list

    # Selection is the first stage rejecter and image cropping
    class Selection:
        def __init__(self, svm_model_path, abcnet_model_path, device='cpu'):
            self.clf = pickle.load(open(svm_model_path, 'rb'))
            self.cfg = get_cfg()
            self.abcnet = ABCNet_inf(self.cfg, device, output_stage=0)
            self.abcnet.to(device)
            self.abcnet.eval()
            self.abcnet.load_state_dict(torch.load(abcnet_model_path, map_location='cpu')['model'], strict=False)

        @staticmethod
        def apply_canny(img, sigma=0.33):
            v = np.median(img)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            return cv2.Canny(img, lower, upper)

        @staticmethod
        def skip_images(mean_x, mean_y):
            if mean_x > 12000 and mean_y > 15000:
                return True
            else:
                return False

        @staticmethod
        def reject(peaks_x, peaks_y, mean_x, mean_y):
            if (mean_x < 400 and mean_y < 400) or len(peaks_x) <= 1 or len(peaks_y) <= 1:
                return False
            else:
                return True

        def second_stage_rejecter(self, image, device='cpu'):
            if image.shape[0] <= 200 or image.shape[1] <= 200:
                raise ValueError('Image size must greater or larger than 200 * 200')

            # extract cropped images feature
            with torch.no_grad():
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                img = img.unsqueeze(0)
                features = self.abcnet(img)
                if device != 'cpu':
                    features = features[0].cpu().numpy()
                else:
                    features = features[0].numpy()
                features = features.flatten()
            # Get prediction from SVM
            predict = self.clf.predict([features])
            return predict

        def select(self, input_image):
            yuv_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
            canny_gray = self.apply_canny(yuv_img[:, :, 0])
            canny_U = self.apply_canny(yuv_img[:, :, 1])
            canny_V = self.apply_canny(yuv_img[:, :, 2])
            auto = canny_U | canny_V | canny_gray

            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(auto, cv2.MORPH_CLOSE, kernel)
            hist_x = np.sum(closing, axis=0)
            hist_y = np.sum(closing, axis=1)
            mean_x = np.mean(hist_x)
            mean_y = np.mean(hist_y)
            peaks_x, _ = find_peaks(hist_x)
            peaks_y, _ = find_peaks(hist_y)

            # First stage rejecter and cropping + second stage rejecter
            if self.reject(peaks_x, peaks_y, mean_x, mean_y):

                if self.skip_images(mean_x, mean_y):
                    xmin, xmax, ymin, ymax = int(peaks_x[0]), int(peaks_x[-1]), int(peaks_y[0]), int(peaks_y[-1])
                    if (ymax - ymin) <= 200 or (xmax - xmin) <= 200:
                        return None
                    predict = self.second_stage_rejecter(input_image[ymin:ymax, xmin:xmax])
                    if predict == 1:
                        return input_image[ymin:ymax, xmin:xmax]
                    # Rejected by second stage rejecter
                    else:
                        return None
                else:
                    xmin, xmax, ymin, ymax = int(peaks_x[0]), int(peaks_x[-1]), int(peaks_y[0]), int(peaks_y[-1])

                if xmin > 15:
                    xmin -= 15
                if xmax < len(hist_x) - 15:
                    xmax += 15
                if ymin > 15:
                    ymin -= 15
                if ymax < len(hist_y) - 15:
                    ymax += 15

                # Reject images w or h that's less than 200 after cropped
                if (ymax - ymin) <= 200 or (xmax - xmin) <= 200:
                    return None

                predict = self.second_stage_rejecter(input_image[ymin:ymax, xmin:xmax])
                if predict == 1:
                    return input_image[ymin:ymax, xmin:xmax]
                # Rejected by second stage rejecter
                else:
                    return None

            # Rejected by first stage rejecter
            else:
                return None

    class Detection:
        def __init__(self, part1_path, part2_path, part3_path, rejector_path, fc_path, thrshold):
            self.model = FOTSModel_q(
                part1_path=part1_path,
                part2_path=part2_path,
                part3_path=part3_path,
                rejector_path=rejector_path,
                fc_path=fc_path,
                thrshold=thrshold)

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

        def prepare_inputs(self, image):
            scale_y = 270 / min(image.shape[0], image.shape[1])  # 1248 # 704
            scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale_y, fy=scale_y, interpolation=cv2.INTER_CUBIC)

            scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
            scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            image_tensor = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()
            return image_tensor, scale_y

        def detect(self, input_image):
            inputs, scale = self.prepare_inputs(input_image)
            #             boxes, scores, _ = self.model(inputs)
            confidence, distances, angle = self.model(inputs)

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

    class Recognition:
        def __init__(self, model_path, model_cnn_path, model_ew0_path, model_ew1_path, charset=None):
            #             self.rcg_module = torch.jit.load(model_path)
            self.rcg_module = crnn.CRNN(32, 1, 37, 256)
            self.rcg_module.eval()
            self.rcg_module.load_state_dict(load_multi(model_path), strict=False)

            self.rcg_module.rnn = torch.quantization.quantize_dynamic(self.rcg_module.rnn, {nn.LSTM}, dtype=torch.qint8)

            for idx, m in enumerate(self.rcg_module.rnn.children()):
                m.set_wrap()
            self.rcg_module.cnn = torch.jit.load(model_cnn_path)

            for idx, m in enumerate(self.rcg_module.rnn.children()):
                if idx == 0:
                    m.embedding_w = torch.jit.load(model_ew0_path)
                else:
                    m.embedding_w = torch.jit.load(model_ew1_path)

            self.alphabet = charset
            self.converter = utils.strLabelConverter(self.alphabet)
            self.transformer = utils.resizeNormalize((100, 32))

        #             self.image_height = image_height
        #             self.hs_factor = hs_factor
        #             self.charset = charset

        def prepare_inputs(self, input_image, box):
            # Find homography transform and apply
            # image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            image = input_image
            box = box.astype(int)
            b_w = int(np.linalg.norm(box[0] - box[1]))
            b_h = int(np.linalg.norm(box[1] - box[2]))
            trgt_box = np.array(
                [[0, 0], [b_w, 0], [b_w, b_h], [0, b_h]], dtype="float32")
            # Apply transform
            xfm_mat = cv2.getPerspectiveTransform(
                np.array(box, dtype="float32"), trgt_box)
            text_patch = cv2.warpPerspective(image, xfm_mat, (b_w, b_h))
            # cv2.imwrite('../{}.jpg'.format(random.randint(0, 20)), cv2.cvtColor(text_patch, cv2.COLOR_RGB2BGR))
            text_patch = text_patch.transpose(1, 0, 2)

            text_patch = cv2.cvtColor(text_patch, cv2.COLOR_RGB2GRAY)
            text_patch = text_patch.transpose(1, 0)
            text_patch = self.transformer(text_patch)
            text_patch = text_patch.view(1, *text_patch.size())
            text_patch = Variable(text_patch)
            return text_patch

        def recognize(self, input_image, box):
            # Setup inputs
            inputs = self.prepare_inputs(input_image, box)

            preds = self.rcg_module(inputs)
            #             symbols, score = self.rcg_module(inputs)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)

            return sim_pred


def load_multi(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict
