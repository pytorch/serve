"""
Decorator for timing handler methods

Use this decorator to compute the execution time for your preprocesss, inference and
postprocess methods.
By default this feature is not enabled.

To enable this, add the following section in your model-config.yaml file

handler:
  profile: true

An example of running benchmarks with the profiling enabled is in
https://github.com/pytorch/serve/tree/master/examples/benchmarking/resnet50

"""

import time

import torch


def timed(func):
    def wrap_func(self, *args, **kwargs):
        # Measure time if config specified in model_yaml_config
        if (
            self.context
            and "handler" in self.context.model_yaml_config
            and "profile" in self.context.model_yaml_config["handler"]
        ):
            if self.context.model_yaml_config["handler"]["profile"]:
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
                metrics = self.context.metrics
                metrics.add_time("ts_handler_" + func.__name__, duration)
            else:
                # If profile config specified in model_yaml_config is False
                result = func(self, *args, **kwargs)
        else:
            # If no profile config specified in model_yaml_config
            result = func(self, *args, **kwargs)

        return result

    return wrap_func
