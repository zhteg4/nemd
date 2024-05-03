import numba
import inspect
import functools
from nemd import environutils

NOPYTHON = 'nopython'
PARALLEL = 'parallel'
CACHE = 'cache'
NOPYTHON_MODE = environutils.NOPYTHON_MODE
CACHE_MODE = environutils.CACHE_MODE


def jit(*args, **kwargs):
    """
    Decorate a function using numba.jit.

    See 'Writing a decorator that works with and without parameter' in
    https://stackoverflow.com/questions/5929107/decorators-with-parameters

    :return 'function': decorated function
    """

    def _decorator(func):
        pmode = environutils.get_python_mode()
        kwargs[NOPYTHON] = kwargs.get(NOPYTHON, pmode
                                      in [NOPYTHON_MODE, CACHE_MODE])
        if kwargs[NOPYTHON]:
            kwargs[CACHE] = kwargs.get(CACHE, pmode == CACHE_MODE)
            nargs = [] if args and callable(args[0]) else args
            func = numba.jit(func, *nargs, **kwargs)
        if NOPYTHON in inspect.signature(func).parameters:
            func = functools.partial(func, nopython=kwargs[NOPYTHON])

        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            return func(*func_args, **func_kwargs)

        return wrapper

    return _decorator(args[0]) if args and callable(args[0]) else _decorator
