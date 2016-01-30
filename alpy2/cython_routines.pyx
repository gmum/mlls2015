from cpython cimport array
cimport numpy as np
cdef score_cython_cpdef(array.array ids, np.ndarray[np.float32_t, ndim=2] distance_cache,
            np.ndarray[np.float32_t, ndim=1] base_scores, c, int n):
    cdef float score_u, score_b
    cdef int id1, id2
    cdef float* base_scores_ptr = &base_scores[0]
    cdef float* distance_cache_ptr = &distance_cache[0,0]
    score_u = 0
    score_b = 0
    for i in range(n):
        id1 = ids.data.as_ints[i]
        score_u += base_scores_ptr[id1]
        for j in range(n):
            id2 = ids.data.as_ints[j]
            if id1 > id2:
                score_b += distance_cache_ptr[id1 + id2*n]
    return (1-c) * score_u / n + c/((n * (n - 1)) / 2.0) * score_b

def score_cython(ids, np.ndarray[np.float32_t, ndim=2] distance_cache,
            np.ndarray[np.float32_t, ndim=1] base_scores, c):
    return score_cython_cpdef(array.array('i', ids), distance_cache, base_scores, c, len(ids))