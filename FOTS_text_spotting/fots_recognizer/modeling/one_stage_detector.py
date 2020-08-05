from torch import nn

import logging
import cv2

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.utils.logger import log_first_n
from typing import Tuple

from detectron2.config.config import configurable

from detectron2.structures import Instances, Boxes
from detectron2.modeling.roi_heads.roi_heads import build_roi_heads
from detectron2.layers.shape_spec import ShapeSpec

import torch
import numpy as np
from fots_detector.model import FOTSModel
from fots_detector.modules.parse_polys import parse_polys


class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float]
    ):
        super().__init__()
        self.roi_heads = roi_heads
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))

    @classmethod
    def from_config(cls, cfg):
        out_shp = {
            "p2": ShapeSpec(
                channels=256, stride=4
            )
        }
        return {
            "roi_heads": build_roi_heads(cfg, out_shp),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
        }

    @property
    def device(self):
        return self.pixel_mean.device

@META_ARCH_REGISTRY.register()
class OneStageRCNN(GeneralizedRCNN):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.to(self.device)
        self.cfg = cfg
        net = FOTSModel()
        checkpoint = torch.load("/data/sairahul5223/FOTS_text_spotting/fots_detector.pt")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net = net.eval().cuda()
        if self.training:
            for param in net.parameters():
                param.requires_grad = False
        self.net = net

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    @staticmethod
    def d2_postprocesss(results, output_height, output_width):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """

        # Converts integer tensors to float temporaries
        #   to ensure true division is performed when
        #   computing scale_x and scale_y.
        if isinstance(output_width, torch.Tensor):
            output_width_tmp = output_width.float()
        else:
            output_width_tmp = output_width

        if isinstance(output_height, torch.Tensor):
            output_height_tmp = output_height.float()
        else:
            output_height_tmp = output_height

        scale_x, scale_y = (
            output_width_tmp / results.image_size[1],
            output_height_tmp / results.image_size[0],
        )
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        return results

    @staticmethod
    def detector_postprocess(results, output_height, output_width):
        """
        In addition to the post processing of detectron2, we add scalign for
        bezier control points.
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        results = OneStageRCNN.d2_postprocesss(results, output_height, output_width)
        try:
            # scale bezier points
            if results.has("beziers"):
                beziers = results.beziers
                # scale and clip in place
                beziers[:, 0::2] *= scale_x
                beziers[:, 1::2] *= scale_y
                h, w = results.image_size
                beziers[:, 0].clamp_(min=0, max=w)
                beziers[:, 1].clamp_(min=0, max=h)
                beziers[:, 6].clamp_(min=0, max=w)
                beziers[:, 7].clamp_(min=0, max=h)
                beziers[:, 8].clamp_(min=0, max=w)
                beziers[:, 9].clamp_(min=0, max=h)
                beziers[:, 14].clamp_(min=0, max=w)
                beziers[:, 15].clamp_(min=0, max=h)
        except:
            print("no detection")
        return results

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        fots_image = cv2.imread(batched_inputs[0]["file_name"])
        scale_x = 640 / fots_image.shape[1]  # 2240 # 1280
        scale_y = 512 / fots_image.shape[0]  # 1248 # 704
        scaled_image = cv2.resize(fots_image, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
        orig_scaled_image = scaled_image.copy()

        scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
        scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        fots_image_input = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()
        images = []
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        scaled_height = 512 / gt_instances[0].image_size[0]
        scaled_width = 640 / gt_instances[0].image_size[1]
        gt_bezier = gt_instances[0].get("beziers")

        gt_bezier[:, ::2] = gt_bezier[:, ::2] * scaled_width
        gt_bezier[:, 1::2] = gt_bezier[:, 1::2] * scaled_height
        gt_instances[0].set("beziers", gt_bezier)
        gt_instances[0].get("gt_boxes").tensor[:, ::2] = gt_instances[0].get("gt_boxes").tensor[:, ::2] * scaled_width
        gt_instances[0].get("gt_boxes").tensor[:, 1::2] = gt_instances[0].get("gt_boxes").tensor[:, 1::2] * scaled_height
        gt_instances[0]._image_size = (512, 640)

        with torch.no_grad():
            confidence, distances, angle, final, d2 = self.net(fots_image_input.cuda())
            final = final.cuda()
            # Upsample feature map
            # final = self.upsample(self.upsample(final))
            # final = self.upsample(final)

        _, detector_losses = self.roi_heads(final, [], gt_instances)

        losses = {}
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        old_height = batched_inputs[0]["height"]
        old_width = batched_inputs[0]["width"]


        with torch.no_grad():
            confidence, distances, angle, final, d2 = self.net(batched_inputs[0]["image"].cuda())
            final = final.cuda()
            # Upsample feature map
            # final = self.upsample(self.upsample(final))
            #
            confidence = torch.sigmoid(confidence).squeeze().data.cpu().numpy()
            distances = distances.squeeze().data.cpu().numpy()
            angle = angle.squeeze().data.cpu().numpy()
            fots_polys = parse_polys(confidence, distances, angle, 0.95, 0.3)
            fots_polys = fots_polys.tolist()
        polys = fots_polys
        i_bboxes = []
        i_locs = []
        i_scores = []
        i_classes = []
        i_top_feats = []
        cnt = 0
        for poly in polys:
            x1, y1 = poly[0], poly[1]
            x4, y4 = poly[2], poly[3]
            x2, y2 = x1 + float(x4 - x1) / float(3), y1 + float(y4 - y1) / float(3)
            x3, y3 = x1 + 2 * (float(x4 - x1) / float(3)), y1 + 2 * (float(y4 - y1) / float(3))
            x5, y5 = poly[4], poly[5]
            x8, y8 = poly[6], poly[7]
            x7, y7 = x8 + float(x5 - x8) / float(3), y8 + float(y5 - y8) / float(3)
            x6, y6 = x8 + 2 * (float(x5 - x8) / float(3)), y8 + 2 * (float(y5 - y8) / float(3))
            clip = lambda x, l, u: max(l, min(u, x))
            x1, y1 = clip(x1, 0, old_width - 1), clip(y1, 0, old_height - 1)
            x2, y2 = clip(x2, 0, old_width - 1), clip(y2, 0, old_height - 1)
            x3, y3 = clip(x3, 0, old_width - 1), clip(y3, 0, old_height - 1)
            x4, y4 = clip(x4, 0, old_width - 1), clip(y4, 0, old_height - 1)
            x5, y5 = clip(x5, 0, old_width - 1), clip(y5, 0, old_height - 1)
            x6, y6 = clip(x6, 0, old_width - 1), clip(y6, 0, old_height - 1)
            x7, y7 = clip(x7, 0, old_width - 1), clip(y7, 0, old_height - 1)
            x8, y8 = clip(x8, 0, old_width - 1), clip(y8, 0, old_height - 1)
            xmin = min(x1, x2, x3, x4, x5, x6, x7, x8)
            ymin = min(y1, y2, y3, y4, y5, y6, y7, y8)
            xmax = max(x1, x2, x3, x4, x5, x6, x7, x8)
            ymax = max(y1, y2, y3, y4, y5, y6, y7, y8)
            xmid = float(xmin + ((xmax - xmin) / 2))
            ymid = float(ymin + ((ymax - ymin) / 2))
            i_bboxes.append([xmin, ymin, xmax, ymax])
            i_locs.append([xmid, ymid])
            i_scores.append(0.9)
            cnt = cnt + 1
            i_classes.append(0)
            i_top_feats.append([float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4), float(x5), float(y5), float(x6), float(y6), float(x7), float(y7), float(x8), float(y8)])

        i_bboxes = np.array([np.array(xi) for xi in i_bboxes])
        i_bboxes = torch.from_numpy(i_bboxes).cuda()
        i_locs = np.array([np.array(xi) for xi in i_locs])
        i_locs = torch.from_numpy(i_locs).cuda()
        i_scores = np.array([np.array(xi) for xi in i_scores])
        i_scores = torch.from_numpy(i_scores).cuda()
        i_classes = np.array([np.array(xi) for xi in i_classes])
        i_classes = torch.from_numpy(i_classes).cuda()
        i_top_feats = np.array([np.array(xi) for xi in i_top_feats])
        i_top_feats = torch.from_numpy(i_top_feats).cuda()
        gt_instance = Instances((old_height, old_width))
        gt_instance.pred_boxes = Boxes(i_bboxes)
        gt_instance.scores = i_scores
        gt_instance.pred_classes = i_classes
        gt_instance.locations = i_locs
        gt_instance.top_feat = i_top_feats

        results, labels = self.roi_heads(final, [gt_instance], None)
        return OneStageRCNN._postprocess(results, labels, batched_inputs, [(288, 480)])

    @staticmethod
    def _postprocess(instances, labels, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = OneStageRCNN.detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        ret = [[processed_results, labels]]
        return ret