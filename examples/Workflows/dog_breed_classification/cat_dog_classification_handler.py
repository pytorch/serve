from ts.torch_handler.image_classifier import ImageClassifier
import base64

labels = ['cat', 'dog']

class CatDogClassifier(ImageClassifier):
    def preprocess(self, data):
        # Base64 encode the image to avoid the framework throwing
        # non json encodable errors
        b64_data = []
        for row in data:
            input_data = row.get("data") or row.get("body")
            # Wrap the input data into a format that is expected by the parent
            # preprocessing method
            b64_data.append({"body": base64.b64decode(input_data)})
        return ImageClassifier.preprocess(self, b64_data)

    def postprocess(self, predictions):
        response = []
        for prediction in predictions:
            prediction = prediction.argmax()
            response.append(labels[prediction])
        return response
