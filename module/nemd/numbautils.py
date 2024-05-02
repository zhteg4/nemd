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

        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            pmode = environutils.get_python_mode()
            kwargs[NOPYTHON] = kwargs.get(NOPYTHON, pmode >= NOPYTHON_MODE)
            kwargs[CACHE] = kwargs.get(CACHE, pmode == CACHE_MODE)
            nargs = [] if args and callable(args[0]) else args
            nb_func = numba.jit(func, *nargs, **
                                kwargs) if kwargs[NOPYTHON] else func
            if NOPYTHON in inspect.signature(func).parameters:
                nb_func = functools.partial(nb_func, nopython=kwargs[NOPYTHON])
            return nb_func(*func_args, **func_kwargs)

        return wrapper

    return _decorator(args[0]) if args and callable(args[0]) else _decorator
