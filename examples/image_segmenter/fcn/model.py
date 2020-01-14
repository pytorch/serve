from torchvision.models.segmentation.fcn import FCN
from torchvision.models import resnet
from intermediate_layer_getter import IntermediateLayerGetter
from fcn import FCNHead
"""
NN definition for fcn_resnet101 i.e. FCN with restnet 101 as backend
"""


class FCNImageSegmenter(FCN):
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

    def load_state_dict(self, state_dict, strict=True):
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        # Credit - https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py#def _load_state_dict()
        import re
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        return super(FCNImageSegmenter, self).load_state_dict(state_dict, strict)