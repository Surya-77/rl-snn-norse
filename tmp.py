from concurrent.futures import ThreadPoolExecutor
import numpy as np


def parallel_worker(in_l, out_l, iter_i):
    print(f"Worker at {in_l[iter_i]}")
    out_l[iter_i] = in_l[iter_i] * iter_i
    return None

if __name__ == "__main__":
    a = list(np.arange(20))
    b = list(np.zeros(20))
    print(f"Input  List: {a}")
    print(f"Output List: {b}")
    tmp_fn = lambda in_list: lambda out_list: lambda iter_list: parallel_worker(in_list, out_list, iter_list)
    with ThreadPoolExecutor(max_workers=20) as executor:
        tmp_fn = tmp_fn(a)
        tmp_fn = tmp_fn(b)
        executor.map(tmp_fn, range(20))
    print(f"Input  List: {a}")
    print(f"Output List: {b}")

