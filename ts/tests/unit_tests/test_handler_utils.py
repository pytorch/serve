from ts.handler_utils.utils import import_class


def test_import_class_no_module_prefix():
    model_class = import_class(
        class_name="transformers.LlamaTokenizer",
    )
    assert "LlamaTokenizer" == model_class.__class__.__name__


def test_import_class_module_prefix():
    model_class = import_class(
        class_name="LlamaTokenizer",
        module_prefix="transformers",
    )
    assert "LlamaTokenizer" == model_class.__class__.__name__
