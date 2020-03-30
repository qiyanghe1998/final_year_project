from vecs_io import *
import csv
from run_pq import parse_args


def topk(data_set, top_ks, top_norms):
    folder = 'output/'
    input_file = folder + '%s_top%s_topnorm%s.csv' % (data_set, str(top_ks), str(top_norms))
    folder_path = folder + data_set
    output_file = './output/%s_top%s_topnorm%s_final.csv' % (data_set, str(top_ks), str(top_norms))

    csvFile = open(input_file, 'r')
    reader = csv.reader(csvFile)

    outputfile = open(output_file, 'w')
    writer = csv.writer(outputfile)

    for line in reader:
        for item in line:
            if len(item) > 0:
                writer.writerow([item])


if __name__ == '__main__':
    dataset = 'imagenet'
    topk = 10
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

    filename = './data/%s/%s_base_sort.fvecs' % (dataset, dataset)

    print("# filename: %s".format(filename))

    X, T, Q, G = loader(dataset, topk, metric, folder='data/')

    print("# sorting")

    num_data = len(X)
    index_list = [[0, 0] for i in range(num_data)]
    #norm = [0 for _ in range(num_data)]
    for i in range(num_data):
        index_list[i] = [np.linalg.norm(X[i]), i]
        #norm[i] = np.linalg.norm(X[i])
    index_list.sort(reverse=True)
    #norm.sort(reverse=True)

    #for i in range(num_data):
    #    if i % 1000 == 999:
    #        print("# %dth the norm is %f and %f and %f\n" % (i, np.linalg.norm(X[index_list[i][1]]), index_list[i][0], norm[i]))

    temp_list = []
    for i in range(num_data):
        temp_list.append(index_list[i][1])

    X = X[temp_list]

    del temp_list
    #for i in range(num_data):
    #    if i != index_list[i][1]:
    #        X[[i, index_list[i][1]]] = X[[index_list[i][1], i]]
            #print("# norm of i X[i] and X[j] are %f and %f\n" % (np.linalg.norm(X[i]), np.linalg.norm(X[index_list[i][1]])))
        # i % 100 == 99:
            #print("swap %d and %d, the current length of X is %d" % (i, index_list[i][1], len(X)))

    #for i in range(num_data):
    #    if i % 1000 == 999:
    #        print("# the norm of %dth data is %f\n" % (i, np.linalg.norm(X[i])))

    print("# begin output")

    fvecs_writer(filename, X)
