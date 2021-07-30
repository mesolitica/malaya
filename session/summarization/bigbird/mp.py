from multiprocessing import Pool
import itertools


def chunks(l, n):
    count = 0
    for i in range(0, len(l), n):
        yield l[i: i + n], count
        count += 1


def multiprocessing(strings, function, cores=16):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()
