from torchvision.models.segmentation.fcn import FCN
from torchvision.models import resnet
from intermediate_layer_getter import IntermediateLayerGetter
from fcn import FCNHead


class FCNImageSegmenter(FCN):
    """
    NN definition for fcn_resnet101 i.e. FCN with restnet 101 as backend
    """

    def __init__(self, num_classes=21, **kwargs):
        backbone = resnet.resnet101( pretrained=True, replace_stride_with_dilation=[False, True, True])
        return_layers = {'layer4': 'out'}
        return_layers['layer3'] = 'aux'
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)
        inplanes = 2048
        classifier = FCNHead(inplanes, num_classes)

        super(FCNImageSegmenter, self).__init__(backbone, classifier, aux_classifier)
