"""
Module for image segmentation default handler
"""
# pylint: disable=E1102,R1721

import io
from PIL import Image
import torch
from torchvision import transforms as T
from .vision_handler import VisionHandler


class ImangeSegmenter(VisionHandler):
    """
    ImangeSegmentor handler class. This handler takes an image
    and returns output as masked image
    Ref - https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
    """

    def __init__(self):
        super(ImangeSegmenter, self).__init__()
        self.image = None

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
        self.image = Image.open(io.BytesIO(image))
        trf = T.Compose([T.Resize(256),
                         T.CenterCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        return  trf(self.image).unsqueeze(0)

    def inference(self, data):
        if torch.cuda.is_available():
            data = data.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(data)['out'][0]

        output_predictions = output.argmax(0)
        return output_predictions

    def postprocess(self, data):
        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(data.byte().cpu().numpy()).resize(self.image.size)
        r.putpalette(colors)

        # convert image to generic jpg format from PIL image for client
        output = io.BytesIO()
        r.convert('RGB').save(output, format='JPEG')
        bin_img_data = output.getvalue()

        return [bin_img_data]

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
