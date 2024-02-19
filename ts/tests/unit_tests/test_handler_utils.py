from ts.handler_utils.utils import import_class


def test_import_class_no_module_prefix():
    model_class = import_class(
        class_name="ts.torch_handler.base_handler.BaseHandler",
    )
    assert "BaseHandler" == model_class.__class__.__name__


def test_import_class_module_prefix():
    model_class = import_class(
        class_name="BaseHandler",
        module_prefix="ts.torch_handler.base_handler",
    )
    assert "BaseHandler" == model_class.__class__.__name__
