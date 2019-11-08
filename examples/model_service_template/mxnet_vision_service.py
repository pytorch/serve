

"""
MXNetVisionService defines a MXNet base vision service
"""
import logging

from mxnet_model_service import MXNetModelService
from mxnet_utils import image, ndarray


class MXNetVisionService(MXNetModelService):
    """
    MXNetVisionService defines a fundamental service for image classification task.
    In preprocess, input image buffer is read to NDArray and resized respect to input
    shape in signature.
    In post process, top-5 labels are returned.
    """

    def preprocess(self, request):
        """
        Decode all input images into ndarray.

        Note: This implementation doesn't properly handle error cases in batch mode,
        If one of the input images is corrupted, all requests in the batch will fail.

        :param request:
        :return:
        """
        img_list = []
        param_name = self.signature['inputs'][0]['data_name']
        input_shape = self.signature['inputs'][0]['data_shape']

        for idx, data in enumerate(request):
            img = data.get(param_name)
            if img is None:
                img = data.get("body")

            if img is None:
                img = data.get("data")

            if img is None or len(img) == 0:
                self.error = "Empty image input"
                return None

            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]

            try:
                img_arr = image.read(img)
            except Exception as e:
                logging.warn(e, exc_info=True)
                self.error = "Corrupted image input"
                return None

            img_arr = image.resize(img_arr, w, h)
            img_arr = image.transform_shape(img_arr)
            img_list.append(img_arr)
        return img_list

    def postprocess(self, data):
        if self.error is not None:
            return [self.error] * self._batch_size

        assert hasattr(self, 'labels'), \
            "Can't find labels attribute. Did you put synset.txt file into " \
            "model archive or manually load class label file in __init__?"
        return [ndarray.top_probability(d, self.labels, top=5) for d in data]


_service = MXNetVisionService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
