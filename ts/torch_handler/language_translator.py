from .text_handler import TextHandler
from torch.autograd import Variable
from torchtext.data.utils import ngrams_iterator
import torch

class LanguageTranslator(TextHandler):
    """
    LanguageTranslator handler class. This handler takes a text (string)
    as input and returns the translated text in destination based on the model.
    """

    def __init__(self):
        super(LanguageTranslator, self).__init__()

    def preprocess(self, data):
        """
         Normalizes the input text for PyTorch model,
         returns an Numpy array
        """

        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")

        ngrams = 2

        #expand contraxtions
        text = self._expand_contactions(text)

        # TODO : use spacy or torchtext's inbuilt tokenizer? `spacy` supports multiple languages.
        text = [tok.text for tok in self._tokenize(text)]

        text = torch.tensor([self.dictionary[token] for token in ngrams_iterator(text, ngrams)])

        return text

    def inference(self, text):
        """
        Predict the class of a text using a trained deep learning model and vocabulary.
        """

        self.model.eval()
        inputs = Variable(text).to(self.device)
        output = self.model.forward(inputs)

        output = output.argmax(1).item() + 1

        if self.mapping:
            output = self.mapping[output]

        return [output]

    def postprocess(self, inference_output):
        return inference_output


_service = LanguageTranslator()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise Exception("The default handler could not translate the input text using the provided model."
                        " Please provide a custom handler in the model archive.")
