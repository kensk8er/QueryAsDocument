"""
Calculate the result for random recommendation.
"""
import math
from random import random
import sys

__author__ = 'kensk8er'


def calculate_ndcg(run_time, precision, k):
    score = 0.
    ndcg = 0.
    opt_ndcg = 0.
    # iterate over run_time
    for i in xrange(run_time):
        for rank in range(1, k + 1):
            relevancy = 1. if random() < precision else 0.

            if rank == 1:
                ndcg += relevancy
                opt_ndcg += 1.
            else:
                ndcg += relevancy / math.log(rank, 2)
                opt_ndcg += 1. / math.log(rank, 2)

        score += ndcg / opt_ndcg

    score /= run_time
    print 'mean NDCG:', score


def calculate_mean_reciprocal_rank(run_time, precision):
    score = 0.

    for i in range(run_time):
        count = 1.
        while True:
            if random() <= precision:
                score += 1. / count
                break
            else:
                count += 1.

    score /= run_time
    print 'MRR:', score

if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1:
        func = args[1]
        func_args = ''
        for i in range(2, len(args)):
            func_args += args[i] + ', '
        func_args = func_args.rstrip()
        func_args = func_args.rstrip(',')
        func = func + '(' + func_args + ')'

        eval(func)
