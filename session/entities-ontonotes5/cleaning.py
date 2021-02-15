from multiprocessing import Pool
import itertools


def chunks(l, r, n):
    for i in range(0, len(l), n):
        yield (l[i : i + n], r[i : i + n])


def multiprocessing(strings, strings_r, function, cores = 16, list_mode = True):
    df_split = chunks(strings, strings_r, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()
    if list_mode:
        return list(itertools.chain(*pooled))
    else:
        return dict(ChainMap(*pooled))
