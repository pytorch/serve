import zipfile
import torch
from io import BytesIO
from torchvision.utils import save_image
from ts.torch_handler.base_handler import BaseHandler

MODELSZIP = "models.zip"
CHECKPOINT = "DCGAN_fashionGen-1d67302.pth"


class ModelHandler(BaseHandler):

    def __init__(self):
        self.initialized = False
        self.map_location = None
        self.device = None
        self.use_gpu = True
        self.store_avg = True
        self.dcgan_model = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")

        self.map_location, self.device, self.use_gpu = \
            ("cuda", torch.device("cuda:"+gpu_id), True) if torch.cuda.is_available() else \
            ("cpu", torch.device("cpu"), False)

        # Extract model architecture source code
        with zipfile.ZipFile(model_dir + "/" + MODELSZIP, "r") as zip_ref:
            zip_ref.extractall(model_dir)

        # Load Model
        from models.DCGAN import DCGAN
        self.dcgan_model = DCGAN(useGPU=self.use_gpu, storeAVG=self.store_avg)
        state_dict = torch.load(model_dir + "/" + CHECKPOINT, map_location=self.map_location)
        self.dcgan_model.load_state_dict(state_dict)

        # ToDo: Test and Hanlde for GPU
        # self.dcgan_model.to(self.device)
        # self.dcgan_model.eval() # Notify all your layers that you are in eval mode

        self.initialized = True

    def preprocess(self, requests):
        preprocessed_data = []
        for req in requests:
            data = req.get("data") if req.get("data") is not None else req.get("body")
            number_of_images = data["number_of_images"]
            random_input, random_labels = self.dcgan_model.buildNoiseData(number_of_images)
            preprocessed_data.append({
                "number_of_images" : number_of_images,
                "input" : random_input
            })
        return preprocessed_data

    def inference(self, preprocessed_data):
        input_batch = torch.cat(tuple(map(lambda d: d["input"], preprocessed_data)), 0)
        with torch.no_grad():
            image_tensor = self.dcgan_model.test(input_batch, getAvG=True, toCPU=True)
        output_batch = torch.split(image_tensor, tuple(map(lambda d: d["number_of_images"], preprocessed_data)))
        return output_batch

    def postprocess(self, output_batch):
        postprocessed_data = []
        for op in output_batch:
            fp = BytesIO()
            save_image(op, fp, format="JPEG")
            postprocessed_data.append(fp.getvalue())
            fp.close()
        return postprocessed_data