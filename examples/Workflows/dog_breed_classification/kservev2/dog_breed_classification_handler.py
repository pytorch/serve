import numpy as np

from ts.torch_handler.image_classifier import ImageClassifier


class DogBreedClassifier(ImageClassifier):
    def preprocess(self, data):
        self.is_dogs = [False] * len(data)
        inp_imgs = []
        for idx, row in enumerate(data):
            cat_dog_response = row.get("cat_dog_classification").decode()
            input_data = row.get("pre_processing").decode()
            if cat_dog_response == "dog":
                self.is_dogs[idx] = True
                # Wrap the input data into a format that is expected by the parent
                # preprocessing method
                inp_imgs.append({"body": input_data})
        if len(inp_imgs) > 0:
            return ImageClassifier.preprocess(self, inp_imgs)

    def inference(self, data, *args, **kwargs):
        if data is not None:
            return ImageClassifier.inference(self, data, *args, **kwargs)

    def postprocess(self, data):
        response = [[]] * len(self.is_dogs)
        if data is None:
            return response
        return [np.array(data).flatten().tolist()]
