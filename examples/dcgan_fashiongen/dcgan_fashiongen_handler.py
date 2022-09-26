import os
import zipfile
import torch
from io import BytesIO
from torchvision.utils import save_image
from ts.torch_handler.base_handler import BaseHandler

MODELSZIP = "models.zip"
CHECKPOINT = "DCGAN_fashionGen.pth"


class ModelHandler(BaseHandler):

    def __init__(self):
        self.initialized = False
        self.map_location = None
        self.device = None
        self.use_gpu = True
        self.store_avg = True
        self.dcgan_model = None
        self.default_number_of_images = 1

    def initialize(self, context):
        """
        Extract the models zip; Take the serialized file and load the model
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")

        self.map_location, self.device, self.use_gpu = \
            ("cuda", torch.device("cuda:"+str(gpu_id)), True) if torch.cuda.is_available() else \
            ("cpu", torch.device("cpu"), False)

        # If not already extracted, Extract model source code
        if not os.path.exists(os.path.join(model_dir, "models")):
            with zipfile.ZipFile(os.path.join(model_dir, MODELSZIP), "r") as zip_ref:
                zip_ref.extractall(model_dir)

        # Load Model
        from models.DCGAN import DCGAN
        self.dcgan_model = DCGAN(useGPU=self.use_gpu, storeAVG=self.store_avg)
        state_dict = torch.load(os.path.join(model_dir, CHECKPOINT), map_location=self.map_location)
        self.dcgan_model.load_state_dict(state_dict)

        self.initialized = True

    def preprocess(self, requests):
        """
        Build noise data by using "number of images" and other "constraints" provided by the end user.
        """
        preprocessed_data = []
        for req in requests:
            data = req.get("data") if req.get("data") is not None else req.get("body", {})

            number_of_images = data.get("number_of_images", self.default_number_of_images)
            labels = {ky: "b'{}'".format(vl) for ky, vl in data.items() if ky not in ["number_of_images"]}

            noise = self.dcgan_model.buildNoiseDataWithConstraints(number_of_images, labels)
            preprocessed_data.append({
                "number_of_images": number_of_images,
                "input": noise
            })
        return preprocessed_data

    def inference(self, preprocessed_data, *args, **kwargs):
        """
        Take the noise data as an input tensor, pass it to the model and collect the output tensor.
        """
        input_batch = torch.cat(tuple(map(lambda d: d["input"], preprocessed_data)), 0)
        with torch.no_grad():
            image_tensor = self.dcgan_model.test(input_batch, getAvG=True, toCPU=True)
        output_batch = torch.split(image_tensor, tuple(map(lambda d: d["number_of_images"], preprocessed_data)))
        return output_batch

    def postprocess(self, output_batch):
        """
        Create an image(jpeg) using the output tensor.
        """
        postprocessed_data = []
        for op in output_batch:
            fp = BytesIO()
            save_image(op, fp, format="JPEG")
            postprocessed_data.append(fp.getvalue())
            fp.close()
        return postprocessed_data
