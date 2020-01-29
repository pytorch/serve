from torchvision.models.vgg import VGG, make_layers, cfgs


class ImageClassifier(VGG):
    def __init__(self):
        super(ImageClassifier, self).__init__(make_layers(cfgs['A'], False), **{'init_weights': False})

