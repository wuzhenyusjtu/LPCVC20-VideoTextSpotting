from typing import Dict
import torch
from torch import nn

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY

from fots_recognizer.layers import conv_with_kaiming_uniform
from ..poolers import TopPooler
from .attn_predictor import ATTPredictor

@ROI_HEADS_REGISTRY.register()
class TextHead(nn.Module):
    """
    TextHead performs text region alignment and recognition.
    
    It is a simplified ROIHeads, only ground truth RoIs are
    used during training.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        Args:
            in_channels (int): number of channels of the input feature
        """
        super(TextHead, self).__init__()
        # fmt: off
        pooler_resolution = cfg.MODEL.BATEXT.POOLER_RESOLUTION
        pooler_scales     = cfg.MODEL.BATEXT.POOLER_SCALES
        sampling_ratio    = cfg.MODEL.BATEXT.SAMPLING_RATIO
        conv_dim          = cfg.MODEL.BATEXT.CONV_DIM
        num_conv          = cfg.MODEL.BATEXT.NUM_CONV
        canonical_size    = cfg.MODEL.BATEXT.CANONICAL_SIZE
        self.in_features  = cfg.MODEL.BATEXT.IN_FEATURES
        self.voc_size     = cfg.MODEL.BATEXT.VOC_SIZE
        self.top_size     = 16
        #
        # fmt: on

        self.pooler = TopPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            canonical_box_size=canonical_size,
            canonical_level=3)

        conv_block = conv_with_kaiming_uniform()
        tower = []
        tower.append(conv_block(conv_dim, conv_dim * 4, 1, 1))
        conv_dim = conv_dim * 4
        for i in range(num_conv):
            tower.append(
                conv_block(conv_dim, conv_dim, 3, 1))
        self.tower = nn.Sequential(*tower)
        
        self.recognizer = ATTPredictor(cfg)


    def forward(self, features, proposals, targets=None):
        """
        see detectron2.modeling.ROIHeads
        """

        features = [features]
        if self.training:
            beziers = [p.beziers for p in targets]
            targets = torch.cat([x.text for x in targets], dim=0)
        else:
            beziers = [p.top_feat for p in proposals]
        bezier_features = self.pooler(features, beziers)
        bezier_features = self.tower(bezier_features)
        # TODO: move this part to recognizer
        if self.training:
            preds, rec_loss = self.recognizer(bezier_features, targets)
            rec_loss *= 0.05
            losses = {'rec_loss': rec_loss}
            return None, losses
        else:
            if bezier_features.size(0) == 0:
                for box in proposals:
                    box.beziers = box.top_feat
                    box.recs = box.top_feat
                return proposals, {}
            preds, _ = self.recognizer(bezier_features, targets)
            start_ind = 0
            for proposals_per_im in proposals:
                end_ind = start_ind + len(proposals_per_im)
                proposals_per_im.recs = preds[start_ind:end_ind]
                proposals_per_im.beziers = proposals_per_im.top_feat
                start_ind = end_ind

            return proposals, preds