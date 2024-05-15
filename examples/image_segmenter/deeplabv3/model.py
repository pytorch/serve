from deeplabv3 import DeepLabHead
from fcn import FCNHead
from intermediate_layer_getter import IntermediateLayerGetter
from torchvision import models
from torchvision.models import resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabV3


class DeepLabV3ImageSegmenter(DeepLabV3):
    """
    NN definition for deeplabv3_resnet101 i.e. DeepLabV3 with resnet 101 as backend
    """

    def __init__(self, num_classes=21, **kwargs):
        backbone = resnet.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1,
            replace_stride_with_dilation=[False, True, True],
        )
        return_layers = {"layer4": "out"}
        return_layers["layer3"] = "aux"
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)
        inplanes = 2048
        classifier = DeepLabHead(inplanes, num_classes)

        super(DeepLabV3ImageSegmenter, self).__init__(
            backbone, classifier, aux_classifier
        )
