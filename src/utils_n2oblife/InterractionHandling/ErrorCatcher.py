import functools

def catch_exception(f):
    """Template of a customed exception catcher

    Args:
        f (function): the function from which catching errors in a class

    Returns:
        any : the return of the function wrapped by the catcher if there is no error
    """
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(f'Caught an exception in {f.__name__}')
    return func