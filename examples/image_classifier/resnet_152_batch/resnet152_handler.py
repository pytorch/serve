import torch

from ts.torch_handler.image_classifier import ImageClassifier


class BatchImageClassifier(ImageClassifier):
    """
    BatchImageClassifier handler class. This handler takes list of images
    and returns a corresponding list of classes
    """

    def preprocess(self, request):
        """
         Scales, crops, and normalizes a PIL image for a PyTorch model,
         returns an Numpy array
        """
        image_tensor = None

        for idx, data in enumerate(request):
            input_image = super(BatchImageClassifier, self).preprocess([data])
            if input_image.shape is not None:
                if image_tensor is None:
                    image_tensor = input_image
                else:
                    image_tensor = torch.cat((image_tensor, input_image), 0)

        return image_tensor

    def postprocess(self, inference_output):
        num_rows, num_cols = inference_output.shape
        output_classes = []
        for i in range(num_rows):
            out = inference_output[i].unsqueeze(0)
            _, y_hat = out.max(1)
            predicted_idx = str(y_hat.item())
            output_classes.append(self.mapping[predicted_idx])
        return output_classes


_service = BatchImageClassifier()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
