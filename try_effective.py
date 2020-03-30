from vecs_io import *
import csv
import time
import heapq as hq


def parse_args():
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--topk', type=int, help='required topk of ground truth')
    parser.add_argument('--topnorm', type=int, help='required top norm data')
    # parser.add_argument('--csv', type=bool, help='request to paint the pdf')
    args = parser.parse_args()
    return args.dataset, args.topk, args.topnorm


def get_threshold(dataset, index_list, top_part, query, top_k):
    temp_value = [0 for i in range(top_part)]
    for i in range(top_part):
        temp_value[i] = np.dot(dataset[index_list[i][1]], query)
    temp_value.sort(reverse=True)
    return temp_value[top_k - 1]
    # return hq.nlargest(top_k, temp_value)[top_k - 1]


def topk(data_set, top_ks, top_norms):
    folder = './data/'
    folder_path = folder + data_set
    base_file = folder_path + '/%s_base.fvecs' % data_set
    query_file = folder_path + '/%s_query_all.fvecs' % data_set
    output_file = './output/%s_top%s_topnorm%s.csv' % (data_set, str(top_ks), str(top_norms))

    outputfile = open(output_file, "w")
    writer = csv.writer(outputfile)

    print("# loading the base data {}, \n".format(base_file))
    X = fvecs_read(base_file)
    print("# loading the queries data {}, \n".format(query_file))
    Q = fvecs_read(query_file)

    num_data = len(X)
    num_query = len(Q)

    print("# sorting %d vectors\n" % num_data)

    last_time = time.time()

    top_part = int(num_data / 100 * top_norms)

    index_list = [[0, 0] for i in range(num_data)]
    for i in range(num_data):
        index_list[i][0] = np.linalg.norm(X[i])
        index_list[i][1] = i
    index_list.sort(reverse=True)

    temp_time = time.time()
    print("# sorting cost time: %ds\n" % int(temp_time - last_time))

    last_time = temp_time
    # for i in range(num_data):
    #    if i % 100000 == 99999:
    #       print("# %dth data with norm %f\n" % (index_list[i][1], np.linalg.norm(X[index_list[i][1]])))

    print("# examing %d queries with %d top part\n" % (num_query, top_part))

    block = (num_query // 1000) * 1

    sum = [0 for i in range(34)]

    for i in range(num_query):
        threshold = get_threshold(X, index_list, top_part, Q[i], top_ks)
        max_value = -10000
        check_num = 0

        left = top_part
        right = num_data
        query_norm = np.linalg.norm(Q[i])
        while left < right - 1:
            mid = (left + right) // 2
            if threshold / query_norm > index_list[mid][0]:
                right = mid
            else:
                left = mid

        for j in range(top_part, left + 1):
            temp_value = np.dot(Q[i], X[index_list[j][1]])
            max_value = max(max_value, temp_value)
            if temp_value > threshold:
                check_num += 1
            # print("# %dth query with threshold %f needs to check %d data in the latter part with max IP %f\n" % (i,
                   # threshold, check_num, max_value), file=outputfile)
        temp = (num_data - top_part - check_num) / (num_data - top_part)
        sum[int(temp * 100 / 3)] += 1

        if i % block == block - 1:
            temp_time = time.time()
            print("# %d queries have finished, cost time: %ds\n" % (i, int(temp_time - last_time)))
            last_time = temp_time
    for i in range(34):
        writer.writerow(['%d%%-%d%%' % (i * 3, min((i + 1) * 3, 100)), (sum[i] / num_query)])


if __name__ == '__main__':
    top_k = 100
    dataset = 'tiny5m'
    top_norm = 10

    import sys

    if len(sys.argv) > 3:
        dataset, top_k, top_norm = parse_args()
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")

    print("# Parameters: dataset = {}, topK = {}, topNorm = {}"
          .format(dataset, top_k, top_norm))

    topk(dataset, top_k, top_norm)