from typing import List


class Summarizer:
    def __init__(self, func, names: List[str], squeeze=True):
        self.func = func
        self.names = names
        self.squeeze = squeeze

    def __call__(self, **vars):
        if self.squeeze:
            vars = {k: v.squeeze() for k, v in vars.items()}
        return self.func(**vars)


def summarystats(*names: str, squeeze=True):
    """Nested decorator that wraps the given function in a `Summarizer` with the given `names`.
    All keyword arguments to the underlying function are assumed to be JAX arrays representing model variables.
    If `squeeze` is true, all variables `squeeze` will be applied`.
    """

    def decorator(func):
        return Summarizer(func, names, squeeze=squeeze)

    return decorator
