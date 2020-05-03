from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.cluster.vq import vq, kmeans2
from numba import cuda

# pq_kernel_code = """
# __global__ void gpu_distance_cal_pq(PQ *pq, float *query, float *result, float *mid_coefficient, int start,
#                                     int end, int num_dim) {
#     int idx = threadIdx.x + blockDim.x * blockIdx.x;
#     if (start + idx < end) {
#         float product_temp = 0;
#         for (int i = 0; i < num_dim; ++i)
#             product_temp += pq.codewords_device[self.compress_code_device[start + idx]][i] * query[i];
#         result[start + idx] += mid_coefficient[start + idx] * product_temp;
#     }
# }
# """


# def gpu_distance_cal(self, query, result, mid_coefficient, start, end):
#     idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
#     if start + idx < end:
#         result[start + idx] = result[start + idx] + mid_coefficient[start + idx] * \
#                               np.dot(self.codewords_device[self.compress_code_device[start + idx]], query)


class PQ(object):
    def __init__(self, M, Ks, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.Ds = None
        self.Dim = -1
        self.number = 0
        self.codewords_device = None
        self.lookup_table = None
        self.codewords_mid_cal_device = None
        self.compress_code_device = None

    def class_message(self):
        return "Subspace PQ, M: {}, Ks : {}, code_dtype: {}".format(self.M, self.Ks, self.code_dtype)

    def fit(self, vecs, iter, open_cuda=False):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        self.Dim = D
        self.number = len(vecs)

        reminder = D % self.M
        quotient = int(D / self.M)
        dims_width = [quotient + 1 if i < reminder else quotient for i in range(self.M)]
        self.Ds = np.cumsum(dims_width)     # prefix sum
        self.Ds = np.insert(self.Ds, 0, 0)  # insert zero at beginning

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, np.max(self.Ds)), dtype=np.float32)
        for m in range(self.M):
            if self.verbose:
                print("#    Training the subspace: {} / {}, {} -> {}".format(m, self.M, self.Ds[m], self.Ds[m+1]))
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            self.codewords[m, :, :self.Ds[m+1] - self.Ds[m]], _ = kmeans2(
                vecs_sub, self.Ks, iter=iter, minit='points')

        if open_cuda:
            self.compress_code_device = cuda.to_device(self.encode(vecs))

        return self

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m]: self.Ds[m+1]]
            codes[:, m], _ = vq(vecs_sub,
                                self.codewords[m, :, :self.Ds[m+1] - self.Ds[m]])

        return codes

    def decode(self, codes):
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.Dim), dtype=np.float32)
        for m in range(self.M):
            vecs[:, self.Ds[m]: self.Ds[m+1]] = self.codewords[m, codes[:, m], :self.Ds[m+1] - self.Ds[m]]

        return vecs

    def compress(self, vecs):
        return self.decode(self.encode(vecs))

    def move_to_gpu(self):
        self.codewords_device = cuda.to_device(self.codewords)

    def cal_lookup_table(self, query):
        self.lookup_table = cuda.shared.array(shape=self.Ks, dtype=np.float32)
        for i in range(self.Ks):
            self.lookup_table[i] = np.dot(self.codewords_device[i], query)

    @cuda.jit
    def gpu_distance_cal(self, query, result, mid_coefficient, start, end):
        idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        if start + idx < end:
            result[start + idx] = result[start + idx] + mid_coefficient[start + idx] * \
                          np.dot(self.codewords_device[self.compress_code_device[start + idx]], query)

    # @cuda.jit
    # def cal(self, threshold, threads_per_block, query):
    #     self.codewords_mid_cal_device = cuda.device_array(self.number, dtype=float)
    #     blocks_per_grid = math.ceil(self.number / threads_per_block)
    #     start_time = time()
    #     gpu_distance_cal[blocks_per_grid, threads_per_block](self.codewords_device, self.compress_code_device, query,
    #                                                          self.codewords_mid_cal_device, self.number)
    #     cuda.synchronize()
    #     print("gpu pq cost {} time".format(str(time() - start_time)))