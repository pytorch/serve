import kfserving
import argparse
import logging
logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)

from TorchserveModel import TorchserveModel
DEFAULT_MODEL_NAME = "model"
DEFAULT_LOCAL_MODEL_DIR = "/tmp/model"
# DEFAULT_MODEL_CLASS_NAME = "mnist"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument('--model_dir', required=True,
                    help='A URI pointer to the model directory')
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
# parser.add_argument('--model_class_name', default=DEFAULT_MODEL_CLASS_NAME,
#                     help='The class name for the model.')
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    #model = PyTorchModel(args.model_name, args.model_class_name, args.model_dir)
    model = TorchserveModel(args.model_name)
    model.load()
    kfserving.KFServer().start([model])