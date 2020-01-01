import torch
from torch.autograd import Variable
from .text_handler import TextHandler
from torchtext.data.utils import ngrams_iterator


class TextClassifier(TextHandler):
    """
    TextClassifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the model vocabulary.
    """

    def __init__(self):
        super(TextClassifier, self).__init__()

    def preprocess(self, data):
        """
         Normalizes the input text for PyTorch model,
         returns an Numpy array
        """

        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")

        ngrams = 2

        # remove html tags
        text = self._remove_html_tags(text)

        # Convert text to all lower case
        text = text.lower()
        text = self._expand_contactions(text)

        # Strip all punctuation from each article
        text = self._remove_punctuations(text)

        # remove the stopwords from input text
        text = self._remove_stopwords(text)

        # remove accented characters
        text = self._remove_accented_characters(text)

        # remove accented characters
        text = self._lemmatize_text(text)

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


_service = TextClassifier()


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
        raise Exception("The default handler could not classify the input text using the provided model."
                        " Please provide a custom handler in the model archive.")
