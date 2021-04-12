from ts.torch_handler.image_classifier import ImageClassifier
import base64

labels = ['cat', 'dog']

class CatDogClassifier(ImageClassifier):
    def preprocess(self, data):
        # HACK: Store a reference to the original input and append it to
        # the output to be used by the second model since there is no way
        # of passing it in a context object
        input_data = data[0].get("data") or data[0].get("body")

        # B64 encode the image to avoid the framework throwing 
        # non json encodable errors
        self.b64_data = base64.b64encode(input_data).decode()
        return ImageClassifier.preprocess(self, data)

    def postprocess(self, prediction):
        prediction = prediction.argmax()
        response = {"output": labels[prediction], "input": self.b64_data}
        return [response]
