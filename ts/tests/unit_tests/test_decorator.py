from ts.utils.util import deprecated


def test_deprecated_func_decorator():
    @deprecated(
        version=1.0,
        replacement="foo1",
    )
    def foo():
        pass

    foo()
