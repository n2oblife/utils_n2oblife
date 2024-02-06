import itertools
import threading
import time
import sys

done = False
#here is the animation
def animate(message="loading", done="Done!"):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write(f'\r{message} ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f'\r{done}')

t = threading.Thread(target=animate, args=[any])
t.start()

#long process here
time.sleep(10)
done = True
time.sleep(0.1)
