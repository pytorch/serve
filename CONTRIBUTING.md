## Contributing to TorchServe
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
           python ts_scripts/install_dependencies.py --environment=dev --cuda=cu121
           ```
            > Supported cuda versions as cu121, cu118, cu117, cu116, cu113, cu111, cu102, cu101, cu92
        - Install `pre-commit` to your Git flow:
            ```bash
            pre-commit install
            ```
        - Run sanity suite
          ```bash
          python torchserve_sanity.py
          ```
    - Run Regression test `python test/regression_tests.py`
    - For running individual test suites refer [code_coverage](docs/code_coverage.md) documentation
    - If you are updating an existing model make sure that performance hasn't degraded by typing running [benchmarks](https://github.com/pytorch/serve/tree/master/benchmarks) on the master branch and your branch and verify there is no performance regression
    - Run `ts_scripts/spellcheck.sh` to fix any typos in your documentation
    - For large changes make sure to run the [automated benchmark suite](https://github.com/pytorch/serve/tree/master/benchmarks) which will run the apache bench tests on several configurations of CUDA and EC2 instances
    - If you need more context on a particular issue, please create raise a ticket on [`TorchServe` GH repo](https://github.com/pytorch/serve/issues/new/choose) or connect to [PyTorch's slack channel](https://pytorch.slack.com/)

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/pytorch/serve.

New features should always be covered by at least one integration test.
For guidance please have a look at our [current suite of pytest tests](https://github.com/pytorch/serve/tree/master/test/pytest) and orient yourself on a test that covers a similar use case as your new feature.
A simplified version of an example test can be found in the [mnist template test](https://github.com/pytorch/serve/blob/master/test/pytest/test_mnist_template.py) which shows how to create a mar file on the fly and register it with TorchServe from within a test.
You can run most tests by simply executing:
```bash
pytest test/pytest/test_mnist_template.py
```
To have a look at the TorchServe and/or test output add `-s` like this:
```bash
pytest -s test/pytest/test_mnist_template.py
```
To run only a subset or a single test from a file use `-k` like this:
```bash
pytest -k  test/pytest/test_mnist_template.py
```

### Install TorchServe for development

If you plan to develop with TorchServe and change some source code, you must install it from source code.

Ensure that you have `python3` installed, and the user has access to the site-packages or `~/.local/bin` is added to the `PATH` environment variable.

Run the following script from the top of the source directory.

NOTE: This script force re-installs `torchserve`, `torch-model-archiver` and `torch-workflow-archiver` if existing installations are found

#### For Debian Based Systems/ MacOS

```
python ./ts_scripts/install_dependencies.py --environment=dev
python ./ts_scripts/install_from_src.py --environment=dev
```

Use `--cuda` flag with `install_dependencies.py` for installing cuda version specific dependencies. Possible values are `cu111`, `cu102`, `cu101`, `cu92`

#### For Windows

Refer to the documentation [here](docs/torchserve_on_win_native.md).

For information about the model archiver, see [detailed documentation](model-archiver/README.md).

### What to Contribute?

### A good first issue
If you've never contributed to TorchServe or OSS before then a great place to start is issues labeled as [`good first issue`](https://github.com/pytorch/serve/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). Bonus points if you personally care about this issue or if it's an issue you filed.

### A good project
If you've used TorchServe in an interesting way, we'd love to feature it in https://github.com/pytorch/serve#-news

#### A new example
The easiest area to contribute a new change to TorchServe is https://github.com/pytorch/serve/tree/master/examples by creating a new handler. Handlers are a simple but powerful paradigm that let you execute arbitrary Python code while serving a model.

For example

The core team developed a single handler to deal with question answering, token classification and sequence classification for HuggingFace models https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py

If you have another use case we'd love to merge it!

As simple as they seem handlers let you do complex stuff like exporting to various runtimes and various people in open source have managed to support [ORT](https://discuss.pytorch.org/t/deploying-onnx-model-with-torchserve/97725/2), [TensorRT](https://github.com/pytorch/serve/issues/1243) and [IPEX](https://github.com/pytorch/serve/tree/master/examples/intel_extension_for_pytorch) without much of the core team's involvement.

```python
class CustomHandler(BaseHandler):
  def initialize(self, ctx):

  def preprocess(self, requests) -> List[Any]:

  def inference(self, input_batch : List[torch.Tensor]) -> List[torch.Tensor]:

  def postprocess(self, inference_output : List[torch.Tensor]) -> List[Any]:

```

The key thing to observe here is that `CustomHandler` is a class so it can hold state that you can save from any of the relevant handler functions. For example you can setup a runtime configuration in `initialize()` and then use it in an `inference()` function. `postprocess()` can return a list of outputs alongside some debug information stored in arbitrary data-structure. `requests` can be anything, video, text or sound data and we've it to this effect in our [multi modal MMF example](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition)

Handlers are incredibly powerful and will let you build meaningful contributions to TorchServe without having to dive deep into the Java internals.

If you are interested in contributing to the internals, we suggest you start here [Internals Guide](docs/internals.md)


### New configurations
To have your custom configurations available in `config.properties` this is a [good educational PR](https://github.com/pytorch/serve/pull/1319) to follow as an example.

All available configurations are set in [ConfigManager.java](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/java/org/pytorch/serve/util/ConfigManager.java) and then can be accessed from handler using [`properties = context.system_properties`](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py). You can also access environment variables directly using `os.environ.get()` and gate behavior based on what that environment variable is.

All of our model optimization work ranging from IPEX to TensorRT work in a similar manner.


### Something more complicated
For something more complicated please open an issue and discuss it with the core team, you can see what our general priorities are here https://github.com/pytorch/serve/projects but if you need a feature urgently we are happy to guide you so you can get it done.
