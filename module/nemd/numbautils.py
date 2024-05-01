import numba
import functools


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
            nargs = [] if args and callable(args[0]) else args
            nb_func = numba.jit(func, *nargs, **kwargs)
            return nb_func(*func_args, **func_kwargs)

        return wrapper

    return _decorator(args[0]) if args and callable(args[0]) else _decorator
