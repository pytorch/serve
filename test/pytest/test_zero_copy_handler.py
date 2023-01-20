from pathlib import Path

import torch
from test_utils import REPO_ROOT

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

EXAMPLE_ROOT_DIR = Path(REPO_ROOT) / "examples" / "zero_copy_model_sharing"


def test_zero_copy_handler(monkeypatch, mocker):
    monkeypatch.syspath_prepend(str(EXAMPLE_ROOT_DIR))
    torch.manual_seed(42)

    # We need to recreate the handler to avoid running into https://github.com/pytorch/text/issues/1849
    def create_and_call_handler(input_text):

        from custom_handler import ZeroCopyModelSharingHandler

        handler = ZeroCopyModelSharingHandler()
        ctx = MockContext(
            model_dir=EXAMPLE_ROOT_DIR.as_posix(),
            model_file=None,
        )

        torch.manual_seed(42)
        handler.initialize(ctx)

        # Try empty string
        x = mocker.Mock(get=lambda _: input_text)

        x = handler.preprocess([x])
        x = handler.inference(x)
        x = handler.postprocess(x)
        return x

    res = create_and_call_handler("Some dogs")

    EXPECTED = [
        "Some dogs can be confused with human beings. Your vet may tell you that a small dog is a small mammal (most small mammals have an average size of 4 feet in length) and, therefore, a pet can weigh less than a normal dog"
    ]

    assert res == EXPECTED
