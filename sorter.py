from pq import *
from multiprocessing import cpu_count
import numba as nb
import math
import tqdm
import time
from pycuda.scan import InclusiveScanKernel


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
def parallel_sort(metric, compressed, Q, X, pq, gpu_index_flat, topk, batch_vecs, threads_per_block, top_norm,
                  norms_sqr=None, prune=False):
    """
    for each q in 'Q', sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param metric: euclid product
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return:
    """

    if not prune:
        rank = np.empty((Q.shape[0], min(131072, compressed.shape[0])), dtype=np.int32)
    else:
        rank = np.empty((Q.shape[0], min(4096, X.shape[0])), dtype=np.int32)

    #p_range = tqdm.tqdm(nb.prange(Q.shape[0]))
    p_range = nb.prange(Q.shape[0])

    if metric == 'product':
        if not prune:
            for i in p_range:
                rank[i, :] = product_arg_sort(Q[i], compressed)
        else:
            for i in p_range:
                rank[i, :] = execute_on_device(pq, X, Q[i], gpu_index_flat, top_norm, topk, batch_vecs,
                                               threads_per_block)

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


@cuda.jit
def gpu_assign(result, mid_coefficient, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        result[idx] = 0
        mid_coefficient[idx] = 1


@cuda.jit
def gpu_mask(result, n, threshold):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        if result[idx] >= threshold:
            result[idx] = 1
        else:
            result[idx] = 0


@cuda.jit
def gpu_map(cal_result, ret, n, ret_start, vecs_start, offset, limit):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        if idx == 0:
            if cal_result[0] == 1:
                ret[ret_start] = vecs_start + offset
        else:
            if cal_result[idx] - cal_result[idx - 1] == 1 and cal_result[idx]:
                ret[ret_start + cal_result[idx] - 1] = vecs_start + offset + idx


def gpu_test(compressed, pq, X, query, gpu_index_flat, top_norm, topk=20, batch_vecs=1024, threads_per_block=256):
    gpu_cal_result = cuda.device_array(batch_vecs, dtype=np.float32)
    mid_coefficient = cuda.device_array(batch_vecs, dtype=np.float32)
    blocks_per_grid = math.ceil(batch_vecs / threads_per_block)

    num_vecs = X.shape[0]

    # norm_filter = np.zeros(num_vecs // batch_vecs)
    # for i in range(num_vecs // batch_vecs):
    #     norm_filter[i] = np.linalg.norm(X[i * batch_vecs - 1])
    # gpu_norm_filter = cuda.device_array(norm_filter, dtype=np.float32)

    num_batch = math.ceil(num_vecs / batch_vecs)

    norm_query = np.linalg.norm(query)
    # get the threshold from faiss
    Q = np.zeros((1, len(query)), dtype=np.float32)
    Q[0] = query
    D, I = gpu_index_flat.search(Q, topk)

    # for j in range(num_batch):
    # get the approximate vectors from codebook
    gpu_assign[threads_per_block, blocks_per_grid](gpu_cal_result, mid_coefficient, batch_vecs)
    cuda.synchronize()

    pq.gpu_distance_cal[threads_per_block, blocks_per_grid](pq, query, gpu_cal_result, mid_coefficient, 0 * batch_vecs,
                                                            min(1 * batch_vecs, num_vecs))
    cuda.synchronize()

    eps = math.pow(10, -10)

    gpu_result = gpu_cal_result.copy_to_host()

    for i in range(batch_vecs):
        temp = np.linalg.norm(compressed[i], query)
        if (gpu_cal_result[i] - temp) > eps:
            print("The {}th vectors error, the gpu result is {}, the compressed result is {}"
                  .format(i, gpu_result[i], temp))


@nb.jit
def execute_on_device(pq, X, query, gpu_index_flat, top_norm, topk=20, batch_vecs=1024, threads_per_block=256):
    gpu_cal_result = cuda.device_array(batch_vecs, dtype=np.float32)
    mid_coefficient = cuda.device_array(batch_vecs, dtype=np.float32)
    blocks_per_grid = math.ceil(batch_vecs / threads_per_block)

    scan_kernel = InclusiveScanKernel(np.int32, "a + b")

    num_vecs = X.shape[0]

    max_length = min(4096, num_vecs)

    gpu_return = cuda.device_array(max_length, dtype=np.int32)
    length = 0

    norm_filter = np.zeros(num_vecs // batch_vecs)
    for i in range(num_vecs // batch_vecs):
        norm_filter[i] = np.linalg.norm(X[i * batch_vecs - 1])
    gpu_norm_filter = cuda.device_array(norm_filter, dtype=np.float32)

    num_batch = math.ceil(num_vecs / batch_vecs)
    cuda.to_device()

    norm_query = np.linalg.norm(query)
    # get the threshold from faiss
    Q = np.zeros((1, len(query)), dtype=np.float32)
    Q[0] = query
    D, I = gpu_index_flat.search(Q, topk)

    for j in range(num_batch):
        # get the approximate vectors from codebook
        gpu_assign[threads_per_block, blocks_per_grid](gpu_cal_result, mid_coefficient, batch_vecs)
        pq.gpu_distance_cal[threads_per_block, blocks_per_grid](query, gpu_cal_result, mid_coefficient, j * batch_vecs,
                                                                min((j + 1) * batch_vecs, num_vecs))
        cuda.synchronize()

        # get the map position for each data
        gpu_mask[threads_per_block, blocks_per_grid](gpu_cal_result, batch_vecs, D[0][-1])
        cuda.synchronize()

        scan_kernel(gpu_cal_result)
        cuda.synchronize()

        if gpu_cal_result[-1] + top_norm + length < max_length:
            gpu_map[threads_per_block, blocks_per_grid](gpu_cal_result, gpu_return, batch_vecs, length,
                                                        j * batch_vecs, top_norm, limit=None)
            cuda.synchronize()
            length += gpu_cal_result[-1]
        else:
            gpu_map[threads_per_block, blocks_per_grid](gpu_cal_result, gpu_return, batch_vecs, length,
                                                        j * batch_vecs, top_norm, limit=
                                                        max_length - length - top_norm - gpu_cal_result[-1])
            cuda.synchronize()
            length = max_length - top_norm
            break
        if j < num_batch - 1:
            if D[-1] > norm_query * gpu_norm_filter[j]:
                break

    # add the candidate in the top norm vectors
    for j in range(topk):
        gpu_return[length + j] = I[j]
    length += topk

    candidate_id = gpu_return[:length].copy_to_host()
    candidate = np.zeros((length, len(X[0])))
    for j in nb.prange(length):
        candidate[j] = X[candidate_id[j]]

    topk_result = product_arg_sort(query, candidate)
    for j in nb.prange(length):
        topk_result = candidate_id[topk_result[j]]

    return topk_result


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


class HybridPruneSorter(object):
    def __init__(self, pq, Q, X, gpu_index_flat, topk, batch_vecs, threads_per_block, metric, top_norm=1000):
        last_time = time.time()
        self.Q = Q
        self.X = X
        self.topK = parallel_sort(metric, None, Q, X, pq, gpu_index_flat, topk, batch_vecs, threads_per_block, top_norm,
                                  prune=True)
        print('# Sorter use {}s time to init'.format(time.time() - last_time))
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


class BatchHybridPruneSorter(object):
    def __init__(self, pq, Q, X, G, Ts, metric, batch_query, gpu_index_flat, topk, batch_vecs, threads_per_block,
                 top_norm):
        self.Q = Q
        self.X = X
        self.recalls = np.zeros(shape=(len(Ts)))
        last_time = time.time()
        for i in range(math.ceil(len(Q) / float(batch_query))):
            q = Q[i*batch_query: (i + 1) * batch_query, :]
            g = G[i*batch_query: (i + 1) * batch_query, :]
            sorter = HybridPruneSorter(pq, q, X, gpu_index_flat, topk, batch_vecs, threads_per_block, metric, top_norm)
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