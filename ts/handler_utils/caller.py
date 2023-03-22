from abc import ABC


class InitCaller(ABC):
    def __call__(self, *args):
        if self._prev:
            self._prev(*args)

        return self._method(*args)


class PipeCaller(ABC):
    def __call__(self, *args):
        if self._prev:
            args = self._prev(*args)
        return self._method(*args)
