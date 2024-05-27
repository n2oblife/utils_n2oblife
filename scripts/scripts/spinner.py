import itertools
import threading
import time
import sys
from tqdm import tqdm
from functools import wraps

def animate(stop_event, messages):
    """Animate a spinner in the console with dynamic loading text.

    Args:
    - stop_event (threading.Event): Event to signal when to stop the spinner.
    - messages (list): List of messages to display in sequence along with the spinner.
    """
    for message, c in zip(itertools.cycle(messages), itertools.cycle(['|', '/', '-', '\\'])):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r{message} {c}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f'\rDone! \n')

def spinner_decorator(messages):
    """Decorator to show a spinner with dynamic loading text while a function is running.

    Args:
    - messages (list): List of messages to display in sequence along with the spinner.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create an event to control the spinner
            stop_event = threading.Event()
            # Start spinner thread
            t = threading.Thread(target=animate, args=(stop_event, messages))
            t.start()
            try:
                # Run the actual function
                result = func(*args, **kwargs)
            finally:
                # Stop the spinner
                stop_event.set()
                # Ensure the spinner thread finishes
                t.join()
            return result
        return wrapper
    return decorator

def dynamic_loading_bar(message="loading", total = 100):
    """Decorator to show a dynamic loading bar while a function is running.

    Args:
    - message (str): Message to display along with the loading bar.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create an event to control the loading bar
            stop_event = threading.Event()
            progress_bar = tqdm(total=total, desc=message, ncols=100, leave=False, bar_format='{desc}: {percentage:3.0f}%|{bar}| {remaining_s}s')

            def update_progress():
                """Update the progress bar while the function runs."""
                for _ in itertools.cycle([0]):
                    if stop_event.is_set():
                        break
                    progress_bar.update(1)
                    time.sleep(0.1)
                    progress_bar.n = (progress_bar.n + 1) % 100
                    progress_bar.last_print_t = time.time()
                    progress_bar.refresh()

            # Start the progress bar thread
            t = threading.Thread(target=update_progress)
            t.start()
            try:
                # Run the actual function
                result = func(*args, **kwargs)
            finally:
                # Stop the loading bar
                stop_event.set()
                # Ensure the loading bar thread finishes
                t.join()
                progress_bar.close()
            return result
        return wrapper
    return decorator

## --- Example of use ---

@spinner_decorator(["Processing", "Still working", "Almost done"])
def long_process():
    # Simulate a long process
    time.sleep(10)

@dynamic_loading_bar("Processing", 300)
def other_long_process():
    # Simulate a long process
    time.sleep(10)

# Call the long process function
long_process()
other_long_process()
