import os

import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.log

from kfserving.kfmodel_repository import KFModelRepository
import requests
import kfserving
import logging
logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)
from .image_transformer import ImageTransformer


class TransformerModelRepository(KFModelRepository):

    def __init__(self, predictor_host:str):
        super().__init__()
        logging.info("ImageTSModelRepo is initialized")
        self.predictor_host = predictor_host
   