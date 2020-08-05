import torch
from torch import nn

from detectron2.modeling.poolers import (
    ROIPooler, convert_boxes_to_pooler_format
)

from fots_recognizer.layers import BezierAlign
from fots_recognizer.structures import Beziers

__all__ = ["TopPooler"]

class TopPooler(ROIPooler):
    """
    ROIPooler with option to assign level by max length. Used by top modules.
    """
    def __init__(self,
                 output_size,
                 scales,
                 sampling_ratio,
                 canonical_box_size=224,
                 canonical_level=4):
        # to reuse the parent initialization, handle unsupported pooler types
        parent_pooler_type = "ROIAlign"
        super().__init__(output_size, scales, sampling_ratio, parent_pooler_type,
                         canonical_box_size=canonical_box_size,
                         canonical_level=canonical_level)
        self.level_poolers = nn.ModuleList(
            BezierAlign(
                output_size, spatial_scale=scale,
                sampling_ratio=sampling_ratio) for scale in scales
        )

    def forward(self, x, box_lists):
        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        if isinstance(box_lists[0], torch.Tensor):
            # TODO: use Beziers for data_mapper
            box_lists = [Beziers(x) for x in box_lists]
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        return self.level_poolers[0](x[0], pooler_fmt_boxes)
