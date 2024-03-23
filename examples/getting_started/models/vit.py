from torchvision.models.vision_transformer import VisionTransformer


class ImageClassifier(VisionTransformer):
    def __init__(self):
        super(ImageClassifier, self).__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
        )
