import pytest

from ts.handler_utils.utils import import_class


def test_import_class_no_module_prefix():
    model_class = import_class(
        class_name="transformers.LlamaTokenizer",
    )
    assert "LlamaTokenizer" == model_class.__name__


def test_import_class_module_prefix():
    model_class = import_class(
        class_name="LlamaTokenizer",
        module_prefix="transformers",
    )
    assert "LlamaTokenizer" == model_class.__name__


def test_import_class_no_module():
    with pytest.raises(ImportError):
        model_class = import_class(
            class_name="LlamaTokenizer",
        )


def test_import_class_no_class():
    with pytest.raises(ImportError):
        model_class = import_class(
            class_name="",
        )
