from vecs_io import *
from pq_residual import *
from sorter import *
from run_pq import execute
from run_pq import parse_args


if __name__ == '__main__':
    dataset = 'imagenet'
    topk = 10
    codebook = 4
    Ks = 256
    metric = 'product'
    group = 4
    prune = True
    top_norm = 5

    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        dataset, topk, codebook, Ks, metric, group, prune, top_norm = parse_args()
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")

    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}, group = {}, prune = {}, "
          "top_norm = {}".format(dataset, topk, codebook, Ks, metric, group, prune, top_norm))

    X, T, Q, G = loader(dataset, topk, metric, folder='data/')
    # pq, rq, or component of norm-pq

    def codebook_group(Ks, codebook):
        pqs = [PQ(M=1, Ks=Ks) for _ in range(codebook - 1)]
        quantizer = ResidualPQ(pqs=pqs)
        return NormPQ(n_percentile=Ks, quantize=quantizer)

    pqs = [codebook_group(Ks, codebook) for _ in range(group)]
    quantizer = ResidualPQ(pqs=pqs)

    #for i in range(len(G)):
    #    print(G[i])

    execute(quantizer, X, T, Q[:100000], G, metric, dataset=dataset, prune=prune, top_norm=top_norm)

    #print(np.shape(X))
    #for i in range(len(X)):
    #    if i % 10000 == 9999:
    #        print("# %dth data's norm is %f\n" % (i, np.linalg.norm(X[i])))
