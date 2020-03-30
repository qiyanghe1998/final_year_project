from pq import *
from multiprocessing import cpu_count
import numba as nb
import math
import tqdm
import time

@nb.jit
def arg_sort(distances):
    top_k = min(131071, len(distances)-1)
    indices = np.argpartition(distances, top_k)[:top_k+1]
    return indices[np.argsort(distances[indices])]


@nb.jit
def product_arg_sort(q, compressed):
    distances = np.dot(compressed, -q)
    return arg_sort(distances)


@nb.jit
def angular_arg_sort(q, compressed, norms_sqr):
    norm_q = np.linalg.norm(q)
    distances = np.dot(compressed, q) / (norm_q * norms_sqr)
    return arg_sort(distances)


@nb.jit
def euclidean_arg_sort(q, compressed):
    distances = np.linalg.norm(q - compressed, axis=1)
    return arg_sort(distances)


@nb.jit
def sign_arg_sort(q, compressed):
    distances = np.empty(len(compressed), dtype=np.int32)
    for i in range(len(compressed)):
        distances[i] = np.count_nonzero((q > 0) != (compressed[i] > 0))
    return arg_sort(distances)


@nb.jit
def euclidean_norm_arg_sort(q, compressed, norms_sqr):
    distances = norms_sqr - 2.0 * np.dot(compressed, q)
    return arg_sort(distances)


@nb.jit
def parallel_sort(metric, compressed, Q, X, norms_sqr=None):
    """
    for each q in 'Q', sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param metric: euclid product
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return:
    """

    rank = np.empty((Q.shape[0], min(131072, compressed.shape[0])), dtype=np.int32)

    #p_range = tqdm.tqdm(nb.prange(Q.shape[0]))
    p_range = nb.prange(Q.shape[0])

    if metric == 'product':
        for i in p_range:
            rank[i, :] = product_arg_sort(Q[i], compressed)
    elif metric == 'angular':
        if norms_sqr is None:
            norms_sqr = np.linalg.norm(compressed, axis=1) ** 2
        for i in p_range:
            rank[i, :] = angular_arg_sort(Q[i], compressed, norms_sqr)
    elif metric == 'euclid_norm':
        if norms_sqr is None:
            norms_sqr = np.linalg.norm(compressed, axis=1) ** 2
        for i in p_range:
            rank[i, :] = euclidean_norm_arg_sort(Q[i], compressed, norms_sqr)
    else:
        for i in p_range:
            rank[i, :] = euclidean_arg_sort(Q[i], compressed)

    return rank


@nb.jit
def true_positives(topK, Q, G, T):
    result = np.empty(shape=(len(Q)))
    for i in nb.prange(len(Q)):
        result[i] = len(np.intersect1d(G[i], topK[i][:T]))
    return result


def get_split(top_part, num_data, query, threshold, dataset):
    left = top_part
    right = num_data
    query_norm = np.linalg.norm(query)
    while left < right - 1:
        mid = (left + right) // 2
        if threshold / query_norm > np.linalg.norm(dataset[mid]):
            right = mid
        else:
            left = mid
    return left + 1


class Sorter(object):
    def __init__(self, compressed, Q, X, metric, norms_sqr=None):
        last_time = time.time()
        self.Q = Q
        self.X = X

        self.topK = parallel_sort(metric, compressed, Q, X, norms_sqr=norms_sqr)
        print('# Sorter use %fs time to init' % (time.time() - last_time))
        del last_time

    def recall(self, G, T):
        t = min(T, len(self.topK[0]))
        return t, self.sum_recall(G, T) / len(self.Q)

    def sum_recall(self, G, T):
        assert len(self.Q) == len(self.topK), "number of query not equals"
        assert len(self.topK) <= len(G), "number of queries should not exceed the number of queries in ground truth"
        true_positive = true_positives(self.topK, self.Q, G, T)
        return np.sum(true_positive) / len(G[0])  # TP / K


class PruneSorter(object):
    def __init__(self, compressed, Q, X, metric, top_norm, norms_sqr=None):
        last_time = time.time()
        self.Q = Q
        self.compressed = compressed
        self.X = X
        self.top_norm = (len(compressed) * top_norm // 100)
        self.topK = parallel_sort(metric, X[:self.top_norm], Q, X)
        #print('PruneSorter use %fs time to init' % (time.time() - last_time))

    def recall(self, G, T):
        t = min(T, len(self.topK[0]))
        return t, self.sum_recall(G, T) / len(self.Q)

    def sum_recall(self, G, T):
        assert len(self.Q) == len(self.topK), "number of query not equals"
        #assert len(self.topK) <= len(G), "number of queries should not exceed the number of queries in ground truth"
        #last_time = time.time()

        t = min(T, len(self.topK[0]))
        index_list = [self.topK[_][:t].tolist() for _ in range(len(self.topK))]



        split = [0 for _ in range(len(self.Q))]
        threshold = [0 for _ in range(len(self.Q))]
        product = [[] for _ in range(len(self.Q))]
        rank = [[] for _ in range(len(self.Q))]

        sum = 0

        for i in nb.prange(len(self.Q)):
            threshold[i] = np.dot(self.Q[i], self.X[self.topK[i][t - 1]])
            split[i] = get_split(self.top_norm, len(self.compressed), self.Q[i], threshold[i], self.compressed)
            #print('# %dth split is %d' % (i, split[i]))
            #if split[i] > self.top_norm:
            product[i] = np.dot(self.compressed[self.top_norm:split[i]], self.Q[i])
            product[i] = np.argwhere(product[i] > threshold[i]) + self.top_norm
            index_list[i].extend(product[i].flatten().tolist())
            #for j in range(self.top_norm, split[i]):
            #    if product[i][j - self.top_norm] > threshold[i]:
            #        index_list[i].append(j)
        del split
        del threshold
        del product

        #print('# prune time: %f' % (time.time() - last_time))
        #last_time = time.time()

        for i in nb.prange(len(self.Q)):
            #print('#%dth index list is like:' % (i))
            #print(index_list[i])
            rank[i] = product_arg_sort(self.Q[i], self.X[index_list[i]])
            rank[i] = [index_list[i][rank[i][j]] for j in range(t)]
            #print('%dth rank list is:' % (i))
            #print(rank[i])

        #print('# re-arrange time: %f' % (time.time() - last_time))
        #last_time = time.time()

        true_positive = true_positives(rank, self.Q, G, t)

        #print('# t = %d evaluate results time: %f' % (t, time.time() - last_time))
        #last_time = time.time()

        return np.sum(true_positive) / len(G[0])  # TP / K


class BatchSorter(object):
    def __init__(self, compressed, Q, X, G, Ts, metric, batch_size, norms_sqr=None):
        self.Q = Q
        self.X = X
        self.recalls = np.zeros(shape=(len(Ts)))
        last_time = time.time()
        for i in range(math.ceil(len(Q) / float(batch_size))):
            q = Q[i*batch_size: (i + 1) * batch_size, :]
            g = G[i*batch_size: (i + 1) * batch_size, :]
            sorter = Sorter(compressed, q, X, metric=metric, norms_sqr=norms_sqr)
            self.recalls[:] = self.recalls[:] + [sorter.sum_recall(g, t) for t in Ts]
            temp_time = time.time()
            print('# one batch time: %f' % (temp_time - last_time))
        self.recalls = self.recalls / len(self.Q)

    def recall(self):
        return self.recalls


class PruneBatchSorter(object):
    def __init__(self, compressed, Q, X, G, Ts, metric, batch_size, top_norm=5, norms_sqr=None):
        self.Q = Q
        self.recalls = np.zeros(shape=(len(Ts)))
        last_time = time.time()
        for i in range(math.ceil(len(Q) / float(batch_size))):
            q = Q[i*batch_size: (i + 1) * batch_size, :]
            g = G[i*batch_size: (i + 1) * batch_size, :]
            sorter = PruneSorter(compressed, q, X, metric=metric, top_norm=top_norm, norms_sqr=norms_sqr)
            self.recalls[:] = self.recalls[:] + [sorter.sum_recall(g, t) for t in Ts]
            temp_time = time.time()
            print('# one batch time: %f' % (temp_time - last_time))
            last_time = temp_time

        self.recalls = self.recalls / len(self.Q)

    def recall(self):
        return self.recalls