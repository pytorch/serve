## Contributing to TorchServe

### What to contribute?
The easiest area to contribute a new change to TorchServe is https://github.com/pytorch/serve/tree/master/examples by creating a new handler. Handlers are a simple but powerful paradigm that let you execute arbitrary Python code while serving a model.

For example

The core team developed a single handler to deal with question answering, token classification and sequence classification for HuggingFace models https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py

If you have another use case we'd love to merge it!

As simple as they seem handlers let you do complex stuff like exporting to various runtimes and various people in open source have managed to support [ORT](https://discuss.pytorch.org/t/deploying-onnx-model-with-torchserve/97725/2), [TensorRT](https://github.com/pytorch/serve/issues/1243) and [IPEX](https://github.com/pytorch/serve/tree/master/examples/intel_extension_for_pytorch) without much of the core team's involvement.

```python
class CustomHandler(BaseHandler):
  def initialize(self, ctx):
  
  def preprocess(self, requests) -> List[]:
    
  def inference(self, input_batch : Tensor) -> Tensor:

  def postprocess(self, inference_output : Tensor) -> List[]:

```

The key thing to observe here is that `CustomHandler` is a class so it can hold state that you can save from any of the relevant handler functions. For example you can setup a runtime configuration in `initialize()` and then use it in an `inference()` function. `postprocess()` can return a list of outputs alongside some debug information stored in arbitrary data-structure. `requests` can be anything, video, text or sound data and we've it to this effect in our [multi modal MMF example](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition)

Handlers are incredibly powerful and will let you build meaningful contributions to TorchServe without having to dive deep into the Java internals.

If you are interested in contributing to the internals, we suggest you start here [Internals Guide](docs/internals.md)

### Merging your code

If you are interested in contributing to TorchServe you'll often need to install it from source and follow some of our guidelines to merge your PRs easily.

Your contributions will fall into two categories:

1. You want to propose a new feature and implement it.
    - Post about your intended feature as an issue, and we will discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    - Search for your issue here: https://github.com/pytorch/serve/issues (look for the "good first issue" tag if you're a first time contributor)
    - Pick an issue and comment on the task that you want to work on this feature.
    - To ensure your changes doesn't break any of the existing features run the sanity suite as follows from serve directory:
        - Install dependencies (if not already installed)
          For CPU
          
          ```bash
          python ts_scripts/install_dependencies.py --environment=dev
          ```
          
         For GPU
           ```bash
           python ts_scripts/install_dependencies.py --environment=dev --cuda=cu102
           ```
            > Supported cuda versions as cu111, cu102, cu101, cu92
       
        - Run sanity suite
          ```bash
          python torchserve_sanity.py
          ```
    - Run Regression test `python test/regression_tests.py`
    - For running individual test suites refer [code_coverage](docs/code_coverage.md) documentation
    - If you are updating an existing model make sure that performance hasn't degraded by running [benchmarks](https://github.com/pytorch/serve/tree/master/benchmarks) on the master branch and your branch and verify there is no performance regression 
    - Run `ts_scripts/spellcheck.sh` to fix any typos in your documentation
    - For large changes make sure to run the [automated benchmark suite](https://github.com/pytorch/serve/tree/master/test/benchmark) which will run the apache bench tests on several configurations of CUDA and EC2 instances
    - If you need more context on a particular issue, please create raise a ticket on [`TorchServe` GH repo](https://github.com/pytorch/serve/issues/new/choose) or connect to [PyTorch's slack channel](https://pytorch.slack.com/)

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/pytorch/serve. Use this [template](pull_request_template.md) when creating a Pull Request.

For more non-technical guidance about how to contribute to PyTorch, see the Contributing Guide.

### Install TorchServe for development

If you plan to develop with TorchServe and change some source code, you must install it from source code.

Ensure that you have `python3` installed, and the user has access to the site-packages or `~/.local/bin` is added to the `PATH` environment variable.

Run the following script from the top of the source directory.

NOTE: This script uninstalls existing `torchserve`, `torch-model-archiver` and `torch-workflow-archiver` installations

#### For Debian Based Systems/ MacOS

```
python ./ts_scripts/install_dependencies.py --environment=dev
python ./ts_scripts/install_from_src.py
```

Use `--cuda` flag with `install_dependencies.py` for installing cuda version specific dependencies. Possible values are `cu111`, `cu102`, `cu101`, `cu92`

#### For Windows

Refer to the documentation [here](docs/torchserve_on_win_native.md).

For information about the model archiver, see [detailed documentation](model-archiver/README.md).

