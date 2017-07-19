import time
import random
import numpy as np

from multiprocessing import Process, Queue, current_process, freeze_support

#
# Function run by worker processes
#

def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)

#
# Functions referenced by tasks
#

def mul(a, b):
    for i0 in range(10):
        np.sqrt(np.linspace(np.random.sample(1), np.sign(np.arccos(0.5)), 1e4))
    return a * b

def plus(a, b):
    for i0 in range(10):
        np.sqrt(np.linspace(np.random.sample(1), np.sign(np.arccos(0.5)), 1e4))
    return a + b

#
#
#

def test():
    NUMBER_OF_PROCESSES = 4
    TASKS1 = [(mul, (i, 7)) for i in range(200000)]
    TASKS2 = [(plus, (i, 8)) for i in range(1)]

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for task in TASKS1:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Unordered results:')
    for i in range(len(TASKS1)):
        done_queue.get()
        # print('\t', done_queue.get())

    # Add more tasks using `put()`
    for task in TASKS2:
        task_queue.put(task)

    # Get and print some more results
    for i in range(len(TASKS2)):
        done_queue.get()
        # print('\t', done_queue.get())

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')


if __name__ == '__main__':
    freeze_support()
    test()


    # from time import time
    # import numpy as np
    # import numexpr as ne
    # import random
    # from multiprocessing import Process, Queue, current_process, freeze_support, Pool, cpu_count
    #
    #
    # def g(x):
    #     y = np.ones(3)
    #     return np.exp(y)
    #
    #
    # def mij(t_u_node, f_nodes):
    #     mypi = np.pi
    #     dxi = ne.evaluate('t_u_node - f_nodes')
    #     dx0 = dxi[:, 0]
    #     dx1 = dxi[:, 1]
    #     dx2 = dxi[:, 2]
    #     dr2 = ne.evaluate('sum(dxi ** 2, axis=1)')
    #     dr1 = ne.evaluate('sqrt(dr2)')
    #     dr3 = ne.evaluate('dr1 * dr2')
    #     temp1 = ne.evaluate('1 / (dr1 * (8 * mypi))')  # 1/r^1
    #     temp2 = ne.evaluate('1 / (dr3 * (8 * mypi))')  # 1/r^3
    #     m00 = ne.evaluate('temp2 * dx0 * dx0 + temp1')
    #     m01 = ne.evaluate('temp2 * dx0 * dx1')
    #     m02 = ne.evaluate('temp2 * dx0 * dx2')
    #     m10 = ne.evaluate('temp2 * dx1 * dx0')
    #     m11 = ne.evaluate('temp2 * dx1 * dx1 + temp1')
    #     m12 = ne.evaluate('temp2 * dx1 * dx2')
    #     m20 = ne.evaluate('temp2 * dx2 * dx0')
    #     m21 = ne.evaluate('temp2 * dx2 * dx1')
    #     m22 = ne.evaluate('temp2 * dx2 * dx2 + temp1')
    #     return m00, m01, m02, m10, m11, m12, m20, m21, m22
    #
    #
    # def m(t_u_node, f_nodes):
    #     mij_args = [(t_u_node, f_nodes)]
    #     mij_args = mij_args * 10000
    #     pool = Pool(cpu_count())
    #     # pool = Pool(1)
    #     mij_list = pool.starmap(mij, mij_args)
    #     return
    #
    #
    # def main_fun():
    #     # p = Pool(4)
    #     # aaa = p.map(g, range(3))
    #     # print(aaa)
    #     t_u_node = np.array((1, 2, 3))
    #     f_nodes = np.asfortranarray(np.random.sample((1000, 3)))
    #     m(t_u_node, f_nodes)
    #
    #
    # if __name__ == '__main__':
    #     t0 = time()
    #     main_fun()
    #     t1 = time()
    #     print('use: %fs' % (t1 - t0))
    #     # print(cpu_count())



    # #
    # # Function run by worker processes
    # #
    # def worker(input, output):
    #     for func, args in iter(input.get, 'STOP'):
    #         result = func(*args)
    #         output.put(result)
    #
    # #
    # # Functions referenced by tasks
    # #
    # def m(t_u_node, f_nodes):
    #     m1 = f_nodes.copy().flatten()
    #     m2 = m1.copy()
    #     m3 = m1.copy()
    #     for i1, t_f_node in enumerate(f_nodes):
    #         # print((f_glbIdx_all[i1 * 3] + 0, f_glbIdx_all[i1 * 3] + 1, f_glbIdx_all[i1 * 3] + 2))
    #         # print((str(obj1), str(obj2), u_glbIdx_all[i0 * 3], f_glbIdx_all[i1 * 3]))
    #         dxi = t_u_node - t_f_node
    #         dr2 = np.sum(dxi ** 2)
    #         dr1 = np.sqrt(dr2)
    #         dr3 = dr1 * dr2
    #         temp1 = 1 / (dr1 * (8 * np.pi))  # 1/r^1
    #         temp2 = 1 / (dr3 * (8 * np.pi))  # 1/r^3
    #         m1[i1 * 3 + 0] = temp2 * dxi[0] * dxi[0] + temp1
    #         m1[i1 * 3 + 1] = temp2 * dxi[0] * dxi[1]
    #         m1[i1 * 3 + 2] = temp2 * dxi[0] * dxi[2]
    #         m2[i1 * 3 + 0] = temp2 * dxi[1] * dxi[0]
    #         m2[i1 * 3 + 1] = temp2 * dxi[1] * dxi[1] + temp1
    #         m2[i1 * 3 + 2] = temp2 * dxi[1] * dxi[2]
    #         m3[i1 * 3 + 0] = temp2 * dxi[2] * dxi[0]
    #         m3[i1 * 3 + 1] = temp2 * dxi[2] * dxi[1]
    #         m3[i1 * 3 + 2] = temp2 * dxi[2] * dxi[2] + temp1
    #     return m1, m2, m3
    #
    #
    # def test():
    #     import multiprocessing
    #     t_u_node = np.array((1,2,3))
    #     f_nodes = np.arange(9).reshape((-1, 3))
    #     TASKS1 = [(m, (t_u_node, f_nodes))]
    #     # Create queues
    #     task_queue = Queue()
    #     done_queue = Queue()
    #
    #     # Start worker processes
    #     for i in range(multiprocessing.cpu_count()):
    #         Process(target=worker, args=(task_queue, done_queue)).start()
    #
    #     # Submit tasks
    #     for task in TASKS1:
    #         task_queue.put(task)
    #
    #     # Get and print results
    #     for i in range(len(TASKS1)):
    #         print(done_queue.get())
    #
    #     # Tell child processes to stop
    #     for i in range(multiprocessing.cpu_count()):
    #         task_queue.put('STOP')
    #
    #
    # if __name__ == '__main__':
    #     t0 = time()
    #     freeze_support()
    #     test()
    #     t1 = time()
    #     print('use: %fs' % (t1 - t0))


    # from functools import partial
    # from itertools import repeat
    # from multiprocessing import Pool, freeze_support
    #
    # def func(a, b):
    #     return a + b
    #
    # def main():
    #     a_args = [i for i in range(10000000)]
    #     second_arg = 1
    #     args = [(a, second_arg) for a in a_args]
    #     with Pool(2) as pool:
    #         L = pool.starmap(func, args)
    #
    # if __name__=="__main__":
    #     freeze_support()
    #     main()
