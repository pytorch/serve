import habana_frameworks.torch.core as htcore  # nopycln: import
import torch

from ts.torch_handler.image_classifier import ImageClassifier


class HPUImageClassifier(ImageClassifier):
    def set_hpu(self):
        self.map_location = "hpu"
        self.device = torch.device(self.map_location)

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        """
        This override of this method allows us to set device to hpu and use the default base_handler without having to modify it.
        """
        model = super()._load_pickled_model(model_dir, model_file, model_pt_path)
        self.set_hpu()
        return model
