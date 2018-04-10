#ifndef HNSW_TESTS
#define HNSW_TESTS

#include <iostream>
#include <algorithm>

#include "utils.h"
#include "hnsw.h"
#include "dumps.h"


float GenerateRandomFloat(int low, int high, bool random_sign);

std::vector<float> GenerateRandomVector(int dim, int low, int high, bool random_sign);

std::vector<Coords> GenerateNRandomVectors(int N, int dim, int low, int high, bool random_sign);


HNSW CreateHNSW(int N, int dim=128, int M=100, int M0=300, int ef_construction=300, float level_multiplier=0.9);


bool TestHNSWSearch(HNSW &hnsw, const Storage &queries);


template<class T>
bool VectorsEqual(const std::vector<T> &first, const std::vector<T> &second);

template<class T>
void PrintVector(const std::string &prefix, const std::vector<T> &vec);


bool TestStorageDump(std::ofstream &ostrm, std::ifstream &istrm, const HNSW &hnsw);


bool TestLevelsDump(std::ofstream &ostrm, std::ifstream &istrm, const HNSW &hnsw);


template<class T>
std::vector<T> CreateVectorFromSet(const std::unordered_set<T> &set);


bool TestHNSWGraphDump(std::ofstream &ostrm, std::ifstream &istrm, const HNSW &hnsw);


bool TestHNSWDump(HNSW &old_hnsw, int K=5, int ef=10);


void RunTests();

#endif // HNSW_TESTS