"""
Handler class for Instruction Embedding models (https://instructor-embedding.github.io/)
"""
import logging

from InstructorEmbedding import INSTRUCTOR

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class InstructorEmbeddingHandler(BaseHandler):
    """
    Handler class for Instruction Embedding models.
    Refer to the README for how to use Instructor models and this handler.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None

    def initialize(self, context):
        properties = context.system_properties
        logger.info("Initializing Instructor Embedding model...")
        model_dir = properties.get("model_dir")
        self.model = INSTRUCTOR(model_dir)
        self.initialized = True

    def handle(self, data, context):
        inputs = data[0].get("body").get("inputs")
        if isinstance(inputs[0], str):
            # single inference
            inputs = [inputs]
        pred_embeddings = self.model.encode(inputs)
        return [pred_embeddings.tolist()]
