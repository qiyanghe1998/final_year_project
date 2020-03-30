from sorter import *
from transformer import *
from vecs_io import loader
import csv
import time


def chunk_compress(pq, vecs):
    chunk_size = 1000000
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs


def get_threshold(dataset, top_part, query, topk):
    temp_value = [0 for i in range(top_part)]
    for i in range(top_part):
        temp_value[i] = np.dot(dataset[i], query)
    temp_value.sort(reverse=True)
    return temp_value[topk - 1]
    # return hq.nlargest(top_k, temp_value)[top_k - 1]


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


def try_effective(X, Q, compressed, dataset, topk=20, topnorm=5):
    output_file = './data/%s/%s_top%s_topnorm%s.csv' % (dataset, dataset, str(topk), str(topnorm))

    outputfile = open(output_file, "w")
    writer = csv.writer(outputfile)

    num_data = len(X)
    num_query = len(Q)

    top_part = int(num_data / 100 * topnorm)

    last_time = time.time()

    print("# examing %d queries with %d top part\n" % (num_query, top_part))

    block = (num_query // 100) * 1

    sum = [0 for i in range(20)]
    sum_real = [0 for i in range(20)]

    for i in range(num_query):
        threshold = get_threshold(compressed, top_part, Q[i], topk)
        threshold_real = get_threshold(X, top_part, Q[i], topk)
        check_num = 0
        check_num_real = 0
        wrong_num = 0
        split = get_split(top_part, num_data, Q[i], threshold, compressed)
        split_real = get_split(top_part, num_data, Q[i], threshold_real, X)

        # print("#%d threshold: %f; real threshold: %f" % (i, threshold, threshold_real))

        for j in range(top_part, split):
            temp_value = np.dot(Q[i], compressed[j])
            if temp_value > threshold:
                check_num += 1
            # print("# %dth query with threshold %f needs to check %d data in the latter part with max IP %f\n" % (i,
            # threshold, check_num, max_value), file=outputfile)
        for j in range(top_part, split_real):
            temp_value = np.dot(Q[i], X[j])
            if temp_value > threshold_real:
                check_num_real += 1

        temp = (num_data - top_part - check_num) / (num_data - top_part)
        sum[min(19, int(temp * 100 / 5))] += 1

        temp = (num_data - top_part - check_num_real) / (num_data - top_part)
        sum_real[min(19, int(temp * 100 / 5))] += 1

        if i % block == block - 1:
            temp_time = time.time()
            print("# %d queries have finished, cost time: %ds\n" % (i, int(temp_time - last_time)))
            last_time = temp_time
    for i in range(19, -1, -1):
        writer.writerow(['compressed', '%d%%-%d%%' % (i * 5, (i + 1) * 5), (sum[i] / num_query)])
        writer.writerow(['real', '%d%%-%d%%' % (i * 5, (i + 1) * 5), (sum_real[i] / num_query)])


def execute(pq, X, T, Q, G, metric, train_size=100000, dataset='netflix', prune=False, top_norm=5):
    np.random.seed(123)
    print("# ranking metric {}".format(metric))
    print("# "+pq.class_message())
    if T is None:
        pq.fit(X[:train_size].astype(dtype=np.float32), iter=20)
    else:
        pq.fit(T.astype(dtype=np.float32), iter=20)

    print('# compress items')
    compressed = chunk_compress(pq, X)
    # print("# evaluate avg error")

    # N, D = np.shape(compressed)
    # print('# the shape of compressed is {} * {}'.format(N, D))

    # sum_error = 0
    # for i in range(len(X)):
    #     sum_error += np.linalg.norm(compressed[i] - X[i])

    # print("expected avg error: {}\n".format(sum_error / len(X)))

    #if prune:
    #    try_effective(X, Q, compressed, dataset)

    print("# sorting items")
    #Ts = [1]
    if prune:
        #Ts = [2 ** i for i in range(2 + int(math.log2(len(X) * top_norm // 100)))]
        Ts = [2 ** i for i in range(2 + 12)]
        recalls = PruneBatchSorter(compressed, Q, X, G, Ts, metric=metric, batch_size=200, top_norm=top_norm).recall()
    else:
        Ts = [2 ** i for i in range(2 + int(math.log2(len(X) * top_norm // 100)))]
        recalls = BatchSorter(compressed, Q, X, G, Ts, metric=metric, batch_size=200).recall()
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))


def parse_args():
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--topk', type=int, help='required topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth')
    parser.add_argument('--num_codebook', type=int, help='number of codebooks in one codebook group')
    parser.add_argument('--Ks', type=int, help='number of centroids in each quantizer')
    parser.add_argument('--num_group', type=int, help='number of codebook groups')
    parser.add_argument('--prune', type=bool, help='use the top norm to prune')
    parser.add_argument('--top_norm', type=int, help='percentage of the top norm')
    args = parser.parse_args()
    return args.dataset, args.topk, args.num_codebook, args.Ks, args.metric, args.num_group, args.prune, args.top_norm


if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'product'
    group = 4

    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        dataset, topk, codebook, Ks, metric, group = parse_args()
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}, group = {}"
          .format(dataset, topk, codebook, Ks, metric, group))

    X, T, Q, G = loader(dataset, topk, metric, folder='data/')
    # pq, rq, or component of norm-pq
    quantizer = PQ(M=codebook, Ks=Ks)
    execute(quantizer, X, T, Q, G, metric)