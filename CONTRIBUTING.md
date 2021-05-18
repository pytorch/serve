## Contributing to TorchServe

If you are interested in contributing to TorchServe, your contributions will fall into two categories:

1. You want to propose a new feature and implement it.
    - Post about your intended feature as an issue, and we will discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    - Search for your issue here: https://github.com/pytorch/serve/issues
    - Pick an issue and comment on the task that you want to work on this feature.
    - To ensure your changes doesn't break any of the existing features run the sanity suite as follows from serve directory:
        - Install dependencies (if not already installed)
          For CPU
          
          ```bash
          python ts_scripts/install_dependencies.py --environment=dev
          ```
          
         For GPU
           ```bash
           python ts_scripts/install_dependencies.py --environment=dev --cuda=cu110
           ```
            > Supported cuda versions as cu110, cu102, cu101, cu92
       
        - Execute sanity suite
          ```bash
          python ./torchserve_sanity.py
          ```
    - For running individual test suites refer [code_coverage](docs/code_coverage.md) documentation
    - If you need more context on a particular issue, please create raise a ticket on [`TorchServe` GH repo](https://github.com/pytorch/serve/issues/new/choose) or connect to [PyTorch's slack channel](https://pytorch.slack.com/)

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/pytorch/serve. Use this [template](pull_request_template.md) when creating a Pull Request.

For more non-technical guidance about how to contribute to PyTorch, see the Contributing Guide.
