from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
)
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FastRCNNConvFCHead,
    _default_anchorgen,
)
from torchvision.models.detection.rpn import RPNHead


class FRCNNObjectDetector(FasterRCNN):
    def __init__(self, num_classes=91, **kwargs):
        trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone = _resnet_fpn_extractor(
            backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d
        )
        rpn_anchor_generator = _default_anchorgen()
        rpn_head = RPNHead(
            backbone.out_channels,
            rpn_anchor_generator.num_anchors_per_location()[0],
            conv_depth=2,
        )
        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7),
            [256, 256, 256, 256],
            [1024],
            norm_layer=nn.BatchNorm2d,
        )
        super(FRCNNObjectDetector, self).__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
            **kwargs
        )
