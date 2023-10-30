import faiss
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import gc

def find_knn(pool, query, use_multi_gpu=False, exact=False, use_gpu_index=0, k=10):
    '''
    This function is used to find knn for query in pool
    output: similarity, indices
    '''
    if not exact:
        d = pool.shape[1]
        quantizer = faiss.IndexFlatIP(d)
        res = faiss.StandardGpuResources()
        index = faiss.IndexIVFPQ(quantizer, d, 30000, 8, 8, faiss.METRIC_INNER_PRODUCT)
        # index = faiss.IndexIVFFlat(quantizer, d, 50000, faiss.METRIC_INNER_PRODUCT)
        # index = faiss.index_factory(d, "PCA64,Flat", faiss.METRIC_INNER_PRODUCT)
        gpu_index = faiss.index_cpu_to_gpu(res, use_gpu_index, index)
        gpu_index.train(pool)
        gpu_index.add(pool)
    elif use_multi_gpu:
        d = pool.shape[1]
        index = faiss.IndexFlatIP(d)
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        gpu_index.add(pool)
    else:
        d = pool.shape[1]
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(d)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(pool)
    print(gpu_index.ntotal)
    similarity, indices = gpu_index.search(query, k*2)
    similarity = get_true_cosine_similarity(pool, query, indices)
    # re rank similarity and indices
    argsort_idx = np.argsort(-similarity, axis=1)
    indices = np.take_along_axis(indices, argsort_idx, axis=1)
    similarity = np.take_along_axis(similarity, argsort_idx, axis=1)
#     indices = indices[np.argsort(-similarity, axis=1)]
#     similarity = similarity[np.argsort(-similarity, axis=1)]

    # return top k
    indices = indices[:, :k]
    similarity = similarity[:, :k]
    print(indices.shape)
    del gpu_index
    gc.collect()
    return similarity, indices

def get_true_cosine_similarity(pool, query, indices):
    similarity = []
    for i in range(indices.shape[0]):
        similarity.append(np.dot(pool[indices[i]], query[i].reshape(-1,\
            1)).reshape(-1))
    return np.array(similarity)
