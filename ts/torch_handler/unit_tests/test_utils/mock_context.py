"""
Mocks for adding model context without loading all of Torchserve
"""

import torch

class MockContext():
    """
    Mock class to replicate the context passed into model initialize
    """
    def __init__(self,
                 model_pt_file='model.pt',
                 model_dir='ts/torch_handler/unit_tests/models/tmp',
                 model_file='model.py',
                 gpu_id='0'):
        self.manifest = {
            'model': {
                'serializedFile': model_pt_file,
            }
        }

        if model_file:
            self.manifest['model']['modelFile'] = model_file

        self.system_properties = {
            'model_dir': model_dir
        }

        if torch.cuda.is_available() and gpu_id:
            self.system_properties['gpu_id'] = gpu_id
