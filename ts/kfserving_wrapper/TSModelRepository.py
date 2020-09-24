import os
from kfserving.kfmodel_repository import KFModelRepository, MODEL_MOUNT_DIRS
from TorchserveModel import TorchserveModel


class TSModelRepository(KFModelRepository):

    def __init__(self, inference_address: str, management_address: str, model_dir: str):
        super().__init__(model_dir)
        print("TSModelRepo is initialized")
        self.inference_address = inference_address
        self.management_address = management_address
        self.model_dir = model_dir

    def load(self, name: str,) -> bool:
        print("TSModelRepository : loading model", name)
        model = TorchserveModel(name, self.inference_address, self.management_address, self. model_dir)
        self.update(model)
        if name in self.models:
            response = model.load_model()
            if response:
                self.update(model)
        else:
            raise KeyError(f"model {name} does not exist")
        return model.ready
    
    def unload(self, name: str):
        print("TSModelRepository : unloading model", name)
        if name in self.models:
            model = self.get_model(name)
            model.unload_model()
            if model.ready == False:
                del self.models[name]
            else :
                raise Exception(f"model {name} is not unloaded")
        else:
            raise KeyError(f"model {name} does not exist")
        