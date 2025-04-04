import io
import json
import torch
import logging
import numpy as np
from os import path
from detectron2.config import get_cfg
from ts.handler_utils.timer import timed
from PIL import Image, UnidentifiedImageError
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
try:
    import pillow_heif
    import pillow_avif
    import pillow_jxl
    # Register openers for extended formats
    pillow_heif.register_heif_opener()
    # For pillow_avif and pillow_jxl, openers are registered upon import
except ImportError as e:
    raise ImportError(
        "Please install 'pillow-heif', 'pillow-avif', and 'pillow-jxl' to handle extended image formats. "
        f"Missing package error: {e}"
    )
########################################################################################################################################
setup_logger()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
########################################################################################################################################
class ModelHandler:
    """
    A base ModelHandler implementation for loading and running Detectron2 models with TorchServe.
    Compatible with both CPU and GPU.
    """
    def __init__(self):
        """
        Initialize the ModelHandler instance.
        """
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False
        self.predictor = None
        self.model_file = "model.pth"
        self.config_file = "config.yaml"
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using GPU for inference.")
        else:
            logger.info("Using CPU for inference.")

    def initialize(self, context):
        """
        Load the model and initialize the predictor.
        Args:
            context (Context): Initial context contains model server system properties.
        """
        logger.info("Initializing model...")

        self._context = context
        self._batch_size = context.system_properties.get("batch_size", 1)
        model_dir = context.system_properties.get("model_dir")
        model_path = path.join(model_dir, self.model_file)
        config_path = path.join(model_dir, self.config_file)
        logger.debug(f"Checking model file: {model_path} exists: {path.exists(model_path)}")
        logger.debug(f"Checking config file: {config_path} exists: {path.exists(config_path)}")
        if not path.exists(model_path):
            error_msg = f"Model file {model_path} does not exist."
            logger.error(error_msg)
            self.error = error_msg
            self.initialized = False
            return
        if not path.exists(config_path):
            error_msg = f"Config file {config_path} does not exist."
            logger.error(error_msg)
            self.error = error_msg
            self.initialized = False
            return
        try:
            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            cfg.MODEL.WEIGHTS = model_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = self.device
            self.predictor = DefaultPredictor(cfg)
            logger.info("Predictor initialized successfully.")
            if self.predictor is None:
                raise RuntimeError("Predictor initialization failed, the predictor is None.")
            self.initialized = True
            logger.info("Model initialization complete.")
        except Exception as e:
            error_msg = "Error during model initialization"
            logger.exception(error_msg)
            self.error = str(e)
            self.initialized = False

    @timed
    def preprocess(self, batch):
        """
        Transform raw input into model input data.

        Args:
            batch (List[Dict]): List of raw requests, should match batch size.

        Returns:
            List[np.ndarray]: List of preprocessed images.
        """
        logger.info(f"Pre-processing started for a batch of {len(batch)}.")

        images = []
        for idx, request in enumerate(batch):
            request_body = request.get("body")
            if request_body is None:
                error_msg = f"Request {idx} does not contain 'body'."
                logger.error(error_msg)
                raise ValueError(error_msg)
            try:
                image_stream = io.BytesIO(request_body)
                try:
                    pil_image = Image.open(image_stream)
                    pil_image = pil_image.convert("RGB")
                    img = np.array(pil_image)
                    img = img[:, :, ::-1]
                except UnidentifiedImageError as e:
                    error_msg = f"Failed to identify image for request {idx}. Error: {e}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                except Exception as e:
                    error_msg = f"Failed to decode image for request {idx}. Error: {e}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                images.append(img)
            except Exception as e:
                logger.exception(f"Error preprocessing request {idx}")
                raise e
        logger.info(f"Pre-processing finished for a batch of {len(batch)}.")
        return images

    @timed
    def inference(self, model_input):
        """
        Perform inference on the model input.

        Args:
            model_input (List[np.ndarray]): List of preprocessed images.

        Returns:
            List[Dict]: List of inference outputs.
        """
        logger.info(f"Inference started for a batch of {len(model_input)}.")

        outputs = []
        for idx, image in enumerate(model_input):
            try:
                logger.debug(f"Processing image {idx}: shape={image.shape}, dtype={image.dtype}")
                output = self.predictor(image)
                outputs.append(output)
            except Exception as e:
                logger.exception(f"Error during inference on image {idx}")
                raise e
        logger.info(f"Inference finished for a batch of {len(model_input)}.")
        return outputs
    @timed
    def postprocess(self, inference_outputs):
        """
        Post-process the inference outputs to a serializable format.

        Args:
            inference_outputs (List[Dict]): List of inference outputs.

        Returns:
            List[str]: List of JSON strings containing predictions.
        """
        logger.info(f"Post-processing for a batch of {len(inference_outputs)}.")
        responses = []
        for idx, output in enumerate(inference_outputs):
            try:
                predictions = output["instances"].to("cpu")
                logger.debug(f"Available prediction fields: {predictions.get_fields().keys()}")
                response = {}
                if predictions.has("pred_classes"):
                    classes = predictions.pred_classes.numpy().tolist()
                    response["classes"] = classes
                if predictions.has("pred_boxes"):
                    boxes = predictions.pred_boxes.tensor.numpy().tolist()
                    response["boxes"] = boxes
                if predictions.has("scores"):
                    scores = predictions.scores.numpy().tolist()
                    response["scores"] = scores
                if predictions.has("pred_masks"):
                    response["masks_present"] = True
                responses.append(json.dumps(response))
            except Exception as e:
                logger.exception(f"Error during post-processing of output {idx}")
                raise e
        logger.info(f"Post-processing finished for a batch of {len(inference_outputs)}.")

        return responses

    @timed
    def handle(self, data, context):
        """
        Entry point for TorchServe to interact with the ModelHandler.

        Args:
            data (List[Dict]): Input data.
            context (Context): Model server context.

        Returns:
            List[str]: List of predictions.
        """
        logger.info("Handling request...")
        if not self.initialized:
            self.initialize(context)
            if not self.initialized:
                error_message = f"Model failed to initialize: {self.error}"
                logger.error(error_message)
                return [error_message]

        if data is None:
            error_message = "No data received for inference."
            logger.error(error_message)
            return [error_message]

        try:
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            output = self.postprocess(model_output)
            return output
        except Exception as e:
            error_message = f"Error in handling request: {str(e)}"
            logger.exception(error_message)
            return [error_message]
########################################################################################################################################
_service = ModelHandler()

def handle(data, context):
    """
    Entry point for TorchServe to interact with the ModelHandler.

    Args:
        data (List[Dict]): Input data.
        context (Context): Model server context.

    Returns:
        List[str]: List of predictions.
    """
    return _service.handle(data, context)
########################################################################################################################################