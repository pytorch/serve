import pytest

from ts.utils.util import deprecated


def test_deprecated_func_decorator():
    @deprecated(
        version=1.0,
        replacement="foo1",
    )
    def foo():
        pass

    with pytest.deprecated_call():
        foo()

    with pytest.warns(PendingDeprecationWarning) as record:
        foo()
        foo()
        assert len(record) == 2
