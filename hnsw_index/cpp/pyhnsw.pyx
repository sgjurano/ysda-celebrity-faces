from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "hnsw.h":
    cdef cppclass HNSW:
        HNSW() except +
        vector[int] KNNSearch(vector[float]&, int, int)


cdef extern from "dumps.h":
    HNSW ReadHNSWFromFile(string storage, string params)


cdef class PyHNSW:
    cdef HNSW _hnsw      # hold a C++ instance which we're wrapping

    def __cinit__(self, string storage, string params):
        self._hnsw = ReadHNSWFromFile(storage, params)

    def knn_search(self, vector[float] coords, int K, int ef):
        return self._hnsw.KNNSearch(coords, K, ef)
