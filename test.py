import ray
import numpy as np
import time


@ray.remote
def return_random() :
    time.sleep(0.1)
    return np.random.random()

ray.init(num_cpus=2)
time.sleep(2.0)
t_start = time.time()

sum = 0
for _ in range(80//8) :
    array = []
    for _ in range(8) :
        array.append(return_random.remote())
    sum += np.sum(ray.get(array))
print(sum)
t_end = time.time()
print("finished!")
print(f"time spent: {t_end-t_start}s")
