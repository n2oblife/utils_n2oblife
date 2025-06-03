import functools
import logging
from typing import Callable, Any, Optional, Union, Type, Tuple


def catch_exception(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    default_return: Any = None,
    reraise: bool = False,
    log_level: str = 'error',
    custom_message: Optional[str] = None,
    return_exception: bool = False
):
    """
    Advanced exception catching decorator with configurable behavior.
    
    Args:
        exceptions: Exception type(s) to catch. Default is Exception (catches all).
        default_return: Value to return when exception is caught. Default is None.
        reraise: Whether to re-raise the exception after logging. Default is False.
        log_level: Logging level ('debug', 'info', 'warning', 'error', 'critical').
        custom_message: Custom message to log. If None, uses default format.
        return_exception: If True, returns the exception object instead of default_return.
    
    Returns:
        Decorator function that wraps the target function with exception handling.
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return f(*args, **kwargs)
            except exceptions as e:
                # Prepare the log message
                if custom_message:
                    message = custom_message.format(
                        func_name=f.__name__,
                        exception=e,
                        exception_type=type(e).__name__
                    )
                else:
                    message = f"Exception in {f.__name__}: {type(e).__name__}: {e}"
                
                # Log the exception at the specified level
                logger = logging.getLogger(__name__)
                log_method = getattr(logger, log_level.lower(), logger.error)
                log_method(message, exc_info=True)
                
                # Handle the return value
                if return_exception:
                    result = e
                else:
                    result = default_return
                
                # Re-raise if requested
                if reraise:
                    raise
                
                return result
        return wrapper
    return decorator


# Simplified versions for common use cases
def catch_all_exceptions(default_return: Any = None, log: bool = True):
    """
    Simple decorator that catches all exceptions and returns a default value.
    
    Args:
        default_return: Value to return when exception occurs.
        log: Whether to log the exception.
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if log:
                    print(f"Exception in {f.__name__}: {type(e).__name__}: {e}")
                return default_return
        return wrapper
    return decorator


def catch_and_log(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    reraise: bool = True
):
    """
    Decorator that logs exceptions and optionally re-raises them.
    
    Args:
        exceptions: Exception type(s) to catch.
        reraise: Whether to re-raise after logging.
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return f(*args, **kwargs)
            except exceptions as e:
                logging.error(
                    f"Exception in {f.__name__}: {type(e).__name__}: {e}",
                    exc_info=True
                )
                if reraise:
                    raise
                return None
        return wrapper
    return decorator


def safe_execute(default_return: Any = None, verbose: bool = False):
    """
    Decorator that ensures a function never raises an exception.
    
    Args:
        default_return: Value to return if exception occurs.
        verbose: Whether to print exception details.
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if verbose:
                    print(f"Safe execution caught exception in {f.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


# Example usage and tests
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Example 1: Original style (improved)
    @catch_all_exceptions(default_return="Error occurred", log=True)
    def divide_simple(a, b):
        return a / b
    
    # Example 2: Advanced configuration
    @catch_exception(
        exceptions=(ZeroDivisionError, TypeError),
        default_return=0,
        log_level='warning',
        custom_message="Math error in {func_name}: {exception}"
    )
    def divide_advanced(a, b):
        return a / b
    
    # Example 3: Catch and re-raise
    @catch_and_log(reraise=False)
    def risky_operation():
        raise ValueError("Something went wrong!")
    
    # Example 4: Safe execution
    @safe_execute(default_return=[], verbose=True)
    def get_data():
        raise ConnectionError("Network error")
    
    # Example 5: Return exception object
    @catch_exception(return_exception=True)
    def might_fail():
        raise RuntimeError("Test error")
    
    # Test the decorators
    print("=== Testing Exception Decorators ===")
    
    print(f"divide_simple(10, 2): {divide_simple(10, 2)}")
    print(f"divide_simple(10, 0): {divide_simple(10, 0)}")
    
    print(f"divide_advanced(10, 2): {divide_advanced(10, 2)}")
    print(f"divide_advanced(10, 0): {divide_advanced(10, 0)}")
    
    print(f"risky_operation(): {risky_operation()}")
    print(f"get_data(): {get_data()}")
    
    exception_result = might_fail()
    print(f"might_fail() returned: {type(exception_result).__name__}: {exception_result}")
    
    
    # Comparison with original decorator
    def original_catch_exception(f):
        """Original implementation (problematic)"""
        @functools.wraps(f)
        def func(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print(f'Caught an exception in {f.__name__}')
                # Problem: No return statement, so returns None implicitly
        return func
    
    @original_catch_exception
    def test_original(x):
        if x == 0:
            raise ValueError("Zero not allowed")
        return x * 2
    
    print("\n=== Original vs Improved ===")
    print(f"Original decorator result: {test_original(5)}")  # Returns None instead of 10
    print(f"Original decorator with error: {test_original(0)}")  # Returns None