# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
from abc import ABC
from csv import get_dialect
import io
import base64
import torch
from PIL import Image
from captum.attr import IntegratedGradients
from ts.torch_handler.base_handler import BaseHandler

from nvidia.dali.fn import image_decoder
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator


### TOREMOVE: Notes to self
###
## There are a few things that make our image handlers slow
## We stack tensors and pass them around in lists instead of creating a batched tensor
# Indexing into it and then passing that whole tensor in one go
## Any for loop in our handler code is code smell

### DALI specific
### We can build a DALI pipeline once in initialization and then pass in input tensors one by one in the same way that
### https://github.com/triton-inference-server/dali_backend/blob/main/client/dali_grpc_client.py
### DALI seems to not mind manually reading input images as lists :?


@pipeline_def(num_threads=4, device_id=0, batch_size=1) #batch_size = 128
def get_dali_pipeline(data):

    for images in data:
        images = fn.decoders.image(images) # This is a CPU only operation
        images = fn.resize(images, resize_x=256, resize_y=256)
        images = fn.decoders.image_random_crop(images, output_type=types.RGB)
        images = fn.crop_mirror_normalize(images,
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                        mirror=fn.random.coin_flip())
    return images

class VisionHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """

    def initialize(self, data):
        pipe = get_dali_pipeline(data)
        # print(data)
        pipe.build()
        output = pipe.run()
        return output

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """

        images = self.initialize(data)
        return images

        ## Allocated a tensor instead of putting images in a list
        # batch_size = len(data)
        # images = torch.empty(batch_size, 3, 224, 224)

        # for i in range(len(data)):
        #     row = data[i]
        #     image = image_decoder(row)
        #     # image = image.resize((256,256))
        #     # image.crop((16,16,240,240))
        #     # image = self.image_processing(image)
        #     images[i] = image
        
        # return images

if __name__ == "__main__":
    handler = VisionHandler()
    data = [torch.randn(3, 224, 224), torch.randn(3,224, 224)]
    output = handler.preprocess(data)
    print(output)

