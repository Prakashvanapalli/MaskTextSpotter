# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, heads):
        super(GeneralizedRCNN, self).__init__()
        self.heads = heads
        self.backbone = build_backbone(cfg)
        for head in heads:
            self.add_module("{}_rpn".format(head), build_rpn(cfg))
            self.add_module("{}_roi_heads".format(head), build_roi_heads(cfg))

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        heads = self.heads

        model_results = {}
        model_losses = {}

        for head in heads:
            rpn = self.__getattr__("{}_rpn".format(head))
            proposals, proposal_losses = rpn(images, features, targets)

            roi_heads = self.__getattr__("{}_roi_heads".format(head))
            if roi_heads:
                x, result, detector_losses = roi_heads(features, proposals, targets)
            else:
                # RPN-only models don't have roi_heads
                x = features
                result = proposals
                detector_losses = {}
            # when running multiple heads results from multiple heads need to be returned 
            model_results = result

            if self.training:
                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)
                model_losses = losses
        if self.training:
            return model_losses

        return model_results
