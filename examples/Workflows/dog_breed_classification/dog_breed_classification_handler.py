from ts.torch_handler.image_classifier import ImageClassifier
import json

class DogBreedClassifier(ImageClassifier):
    def preprocess(self, data):
        self.is_dog = False
        input_data = data[0].get("data") or data[0].get("body")
        input_data = json.loads(input_data)
        if input_data["output"]=="dog":
            self.is_dog = True
            # Wrap the input data into a format that is expected by the parent
            # preprocessing method
            return ImageClassifier.preprocess(self, [{"body": input_data["input"]}])

    def inference(self, data, *args, **kwargs):
        if self.is_dog:
            return ImageClassifier.inference(self, data, *args, **kwargs)

    def postprocess(self, data):
       if self.is_dog:
            return ImageClassifier.postprocess(self, data)
       return ["Not a Dog!"]
