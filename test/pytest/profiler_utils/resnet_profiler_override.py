from ts.torch_handler.image_classifier import ImageClassifier
from torch.profiler import ProfilerActivity, tensorboard_trace_handler


class ResnetHandler(ImageClassifier):
    def __init__(self):
        super().__init__()
        self.profiler_args = {
            "activities" : [ProfilerActivity.CPU],
            "record_shapes": True,
            "on_trace_ready": tensorboard_trace_handler("/tmp/output/resnet-152-batch")
        }
