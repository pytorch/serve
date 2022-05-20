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
import nvidia.dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator


"""
## TOREMOVE: Notes to self
##
# There are a few things that make our image handlers slow
# We stack tensors and pass them around in lists instead of creating a batched tensor
Indexing into it and then passing that whole tensor in one go
# Any for loop in our handler code is code smell. This is a simple change that'll make things much faster

## DALI specific
## We can build a DALI pipeline once in initialization and then pass in input tensors one by one in the same way that
## https://github.com/triton-inference-server/dali_backend/blob/main/client/dali_grpc_client.py
## DALI seems to not mind manually reading input images as lists :?
## DALI only has Ubuntu support so can't drop legacy pipeline and supports Keppler and above
## DALI will shine in CPU bottlenecked siutations so best benchmark is  Amazon EC2 P3.16xlarge,

Initial goal was to only use NVJPEG decoding but that's a DALI op
so will require us to run a full pipeline - do we also need to use their dataa loader?

DALI decode and read assume a JPEG image is being read from disk wheras in our handlers we are given a List[torch.Tensor]
@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='cpu')

    return images, labels


pipe_out = pipe.run()
print(pipe_out)

The output is not a torch.Tensor either
(TensorListCPU(
    [[[[255 255 255]
      [255 255 255]

The graph can be serialized save_graph_to_dot_file(filename, show_tensors=False, show_ids=False, use_colors=False)¶

How to do conversions of DALITensor to a generic torch tensor feed_input(data_node, data, layout=None, cuda_stream=None, use_copy_kernel=False)

DALI supports several different layouts https://docs.nvidia.com/deeplearning/dali/user-guide/docs/data_types.html#interpreting-tensor-layout-strings
So it should give us free support for video

Not all ops are supported on GPU, most are mixed https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html

DALI expects to read input data in its own format, you can customize this behavior with an


# Can copy to a numpy array but then this will be slow
# images.detach().numpy()
# Decoder takes in jpeg images and not torch tensors
# images = fn.decoders.image(images) # This is a CPU only operation

"""

@pipeline_def(num_threads=4, device_id=0, batch_size=2) #batch_size = 128
def get_dali_pipeline(data):
    images = []
    for image in data:


        # images = fn.decoders.image_random_crop(images, output_type=types.RGB)

        # images = fn.resize(images, resize_x=256, resize_y=256)
        # images = fn.crop_mirror_normalize(images,
                                        # mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        # std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                        # mirror=fn.random.coin_flip())
        images.append(image)
        
    return images



class VisionHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """

    def initialize(self, data):
        self.pipe = get_dali_pipeline(data)
        self.pipe.build()
        output = self.pipe.run()

        # print(data)

        return output

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """

        batch_size = 2
        images = self.initialize(data)

        # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_plugin_api.html#nvidia.dali.plugin.pytorch.feed_ndarray
        # pytorch_tensor = torch.empty(batch_size, 3, 224, 224)

        # nvidia.dali.plugin.pytorch.feed_ndarray(images, pytorch_tensor)
        
        output = nvidia.dali.plugin.pytorch.DALIGenericIterator(pipelines=[self.pipe], output_map="hello")


        # print(image)

        return images

        ## Allocated a tensor instead of putting images in a list
        # batch_size = len(data)

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

