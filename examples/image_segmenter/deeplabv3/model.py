from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from torchvision.models import resnet
from intermediate_layer_getter import IntermediateLayerGetter
from deeplabv3 import DeepLabHead
from fcn import FCNHead


class DeepLabV3ImageSegmenter(DeepLabV3):
    """
    NN definition for deeplabv3_resnet101 i.e. DeepLabV3 with resnet 101 as backend
    """

    def __init__(self, num_classes=21, **kwargs):
        backbone = resnet.resnet101(pretrained=True,
                                    replace_stride_with_dilation=[False, True, True])
        return_layers = {'layer4': 'out'}
        return_layers['layer3'] = 'aux'
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)
        inplanes = 2048
        classifier = DeepLabHead(inplanes, num_classes)

        super(DeepLabV3ImageSegmenter, self).__init__(backbone, classifier, aux_classifier)
