from vecs_io import *
from pq_residual import *
from sorter import *
from run_pq import execute
from run_pq import parse_args


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
        dataset, topk, codebook, Ks, metric, group, _, __ = parse_args()
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}, group = {}"
          .format(dataset, topk, codebook, Ks, metric, group))

    X, T, Q, G = loader(dataset, topk, metric, folder='data/')
    # pq, rq, or component of norm-pq
    pqs = [PQ(M=1, Ks=Ks) for _ in range(codebook * group)]
    quantizer = ResidualPQ(pqs=pqs)
    execute(quantizer,  X, T, Q, G, metric)