"""
Module for image segmentation default handler
"""
import io
from PIL import Image
from torchvision import transforms as T
from torch.autograd import Variable
from .vision_handler import VisionHandler


class ImangeSegmenter(VisionHandler):
    """
    ImangeSegmentor handler class. This handler takes an image
    and returns output shape as [CL H W], CL - number of classes, H - height and W - width.
    """

    def __init__(self):
        super(ImangeSegmenter, self).__init__()

    def preprocess(self, data):
        """
        Resize the image to (256 x 256)
        CenterCrop it to (224 x 224)
        Convert it to Tensor - all the values in the image becomes between [0, 1] from [0, 255]
        Normalize it with the Imagenet specific values mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
        And unsqueeze to make [1 x C x H x W] from [C x H x W]
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")
        image = Image.open(io.BytesIO(image))
        trf = T.Compose([T.Resize(256),
                         T.CenterCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        image = trf(image).unsqueeze(0)
        return image

    def inference(self, data):
        # Predict the pixel classes for segmentation
        data = Variable(data).to(self.device)
        pred = self.model(data)['out']
        pred = pred.squeeze().detach().cpu().numpy()
        return [str(pred)]

    def postprocess(self, data):
        return data


_service = ImangeSegmenter()


def handle(data, context):
    """
    Entry point for image segmenter default handler
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise Exception("Please provide a custom handler in the model archive." + e)
