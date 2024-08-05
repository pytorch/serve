import logging

import torch
import transformers
from bs4 import BeautifulSoup as Soup
from hf_custom_embeddings import CustomEmbedding
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import format_document

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class RAGHandler(BaseHandler):
    """
    RAG handler class retrieving documents from a url, encoding & storing in a vector database.
    For a given query, it returns the closest matching documents.
    """

    def __init__(self):
        super(RAGHandler, self).__init__()
        self.vectorstore = None
        self.initialized = False
        self.N = 5

    @torch.inference_mode
    def initialize(self, ctx):
        url = ctx.model_yaml_config["handler"]["url_to_scrape"]
        chunk_size = ctx.model_yaml_config["handler"]["chunk_size"]
        chunk_overlap = ctx.model_yaml_config["handler"]["chunk_overlap"]
        model_path = ctx.model_yaml_config["handler"]["model_path"]

        loader = RecursiveUrlLoader(
            url=url, max_depth=3, extractor=lambda x: Soup(x, "html.parser").text
        )
        docs = loader.load()

        # Split the document into chunks with a specified chunk size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_splits = text_splitter.split_documents(docs)

        # Store the document into a vector store with a specific embedding model
        self.vectorstore = FAISS.from_documents(
            all_splits, CustomEmbedding(model_path=model_path)
        )

    def preprocess(self, requests):
        assert len(requests) == 1, "Expecting batch_size = 1"
        inputs = []
        for request in requests:
            input_text = request.get("data") or request.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            inputs.append(input_text)
        return inputs[0]

    @torch.inference_mode
    def inference(self, data, *args, **kwargs):
        searchDocs = self.vectorstore.similarity_search(data, k=self.N)
        return (searchDocs, data)

    def postprocess(self, data):
        docs, question = data[0], data[1]
        doc_prompt = PromptTemplate.from_template("{page_content}")
        context = ""
        for doc in docs:
            context += f"\n{format_document(doc, doc_prompt)}\n"

        prompt = (
            f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
            f"\n\n{context}\n\nQuestion: {question}"
            f"\nHelpful Answer:"
        )
        return [prompt]
