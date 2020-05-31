import io
import logging
import os
import torch
import torch.nn.functional as F

# Command to build and run the server:
# torch-model-archiver --model-name iris --version 1.0 --model-file examples/iris/iris.py --serialized-file examples/iris/iris.pt --export-path model_store  --handler examples/iris/iris_handler.py --force
# torchserve --start --ncs --foreground --model-store model_store --models iris.mar

logger = logging.getLogger(__name__)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Linear = torch.nn.Linear(4, 3, bias=False)

    def forward(self, x):
        x = self.Linear(x)
        x = F.log_softmax(x, dim=1)
        return x


class IrisClassifier(object):
    """
    IrisClassifier handler class. This handler takes a Nx4 array in json format 
    and returns the class of each feature.
    """

    def __init__(self):
        self.model = None
        self.initialized = False

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        # Read model weights and load them.
        model_pt_path = os.path.join(model_dir, "iris.pt")
        self.state_dict = torch.load(model_pt_path)

        self.model = Net()
        self.model.load_state_dict(self.state_dict)

        logger.debug('Model file {0} loaded successfully'.format(self.model))
        self.initialized = True

    def preprocess(self, data):
        return data

    def inference(self, data):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        import json
        x = json.loads(data[0]["input"].decode('utf8'))
        output = self.model(torch.tensor(x))
        score, predicted = torch.max(output, 1)
        score_, predicted_ = output.topk(3)
        score_ = score_.exp()
        results = []
        for s, p in zip(score_, predicted_):
            results.append([s.tolist(), p.tolist()])
        return [results]
        # The result will be json-fied

    def postprocess(self, inference_output):
        return inference_output


_service = IrisClassifier()


def handle(data, context):
    logger.debug(context)
    
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None

    data = _service.inference(data)
    data = _service.postprocess(data)
    return data
