from joblib import Parallel, delayed
from math import sqrt

with Parallel(n_jobs=4) as parallel:
    accumulator = 0.
    n_iter = 0
    while n_iter < 10000:
        results = parallel(delayed(sqrt)(accumulator + i ** 2)
                           for i in range(5000))
        accumulator += sum(results)  # synchronization barrier
        n_iter += 1
