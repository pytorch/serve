import time

import torch


def timed(func):
    def wrap_func(self, *args, **kwargs):
        # Measure start time
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.time()

        result = func(self, *args, **kwargs)

        # Measure end time
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
        else:
            end = time.time()

        # Measure time taken to execute the function in miliseconds
        if torch.cuda.is_available():
            duration = start.elapsed_time(end)
        else:
            duration = (end - start) * 1000

        # Add metrics for profiling
        if (
            "handler" in self.context.model_yaml_config
            and "profile" in self.context.model_yaml_config["handler"]
        ):
            if self.context.model_yaml_config["handler"]["profile"]:
                metrics = self.context.metrics
                metrics.add_time(
                    self.__class__.__name__ + "_" + func.__name__, duration
                )
        return result

    return wrap_func
