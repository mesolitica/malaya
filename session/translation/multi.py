from multiprocessing import Pool
import itertools
from tqdm import tqdm


def multiprocessing(split, function, cores = 12):
    pool = Pool(cores)
    pooled = pool.starmap(function, split)
    pool.close()
    pool.join()
