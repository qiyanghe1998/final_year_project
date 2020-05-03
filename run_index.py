from pq_residual import *
from run_pq import *
import faiss
import sorter


if __name__ == '__main__':
    dataset = 'imagenet'
    topk = 10
    codebook = 4
    Ks = 256
    metric = 'product'
    group = 4
    prune = True
    top_norm = 5
    batch_vecs = 1024
    batch_query = 200
    threads_per_block = 256

    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        dataset, topk, codebook, Ks, metric, group, prune, top_norm, batch_vecs, batch_query, threads_per_block = \
            parse_args()
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")

    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}, group = {}, prune = {}, "
          "top_norm = {}, batch_vecs = {}, batch_query = {}, threads_per_block = {}"
          .format(dataset, topk, codebook, Ks, metric, group, prune, top_norm, batch_vecs, batch_query,
                  threads_per_block))

    X, T, Q, G = loader(dataset, topk, metric, folder='../data/')
    num_dimen = len(X[0])
    # pq, rq, or component of norm-pq

    def codebook_group(Ks, codebook):
        pqs = [PQ(M=1, Ks=Ks) for _ in range(codebook - 1)]
        quantizer = ResidualPQ(pqs=pqs)
        return NormPQ(n_percentile=Ks, quantize=quantizer)

    pqs = [codebook_group(Ks, codebook) for _ in range(group)]
    quantizer = ResidualPQ(pqs=pqs)

    #for i in range(len(G)):
    #    print(G[i])

    top_num = len(X) * top_norm // 100

    res = faiss.StandardGpuResources()

    # build a flat (CPU) index
    index_flat = faiss.IndexFlat(num_dimen, faiss.METRIC_INNER_PRODUCT)
    # make it into a gpu index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    gpu_index_flat.add(X[:top_num])  # add vectors to the index
    print(gpu_index_flat.ntotal)

    print(prune)

    if prune == 0:
        # execute(quantizer, X, T, Q[:100000], G, metric, dataset=dataset, prune=prune, top_norm=top_norm)
        train(quantizer, X, T, metric, dataset=dataset)

        move_to_gpu(quantizer)

        compressed = get_comrpessed(quantizer, X, T, Q, G, metric, dataset=dataset, prune=prune, top_norm=top_norm)

        gpu_test(compressed, quantizer, X, Q[0], gpu_index_flat, top_norm, topk=20, batch_vecs=1024, threads_per_block=256)
    else:
        move_to_gpu(quantizer)

        execute_on_device(quantizer, Q, X, G, metric, batch_query, gpu_index_flat, topk, batch_vecs, threads_per_block,
                          top_num)

    # topk = 20  # we want to see 4 nearest neighbors
    # D, I = gpu_index_flat.search(Q, topk)  # actual search

    
    # print(I[:20])  # neighbors of the 5 first queries
    # print(I[-20:])  # neighbors of the 5 last queries

    #print(np.shape(X))
    #for i in range(len(X)):
    #    if i % 10000 == 9999:
    #        print("# %dth data's norm is %f\n" % (i, np.linalg.norm(X[i])))
