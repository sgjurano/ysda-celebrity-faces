#include <iostream>
#include <algorithm>

#include "utils.h"
#include "hnsw.h"
#include "dumps.h"
#include "tests.h"


float GenerateRandomFloat(int low, int high, bool random_sign) {
    auto r = std::rand() / static_cast<float>(RAND_MAX);
    if (random_sign && r < 0.5) {
        r *= -1;
    }
    return low + r * static_cast<float>(high - low);
}

std::vector<float> GenerateRandomVector(int dim, int low, int high, bool random_sign) {
    std::vector<float> vec;
    for (int i = 0; i < dim; ++i) {
        vec.push_back(GenerateRandomFloat(low, high, random_sign));
    }
    return vec;
}

std::vector<std::vector<float>> GenerateNRandomVectors(int N, int dim, int low, int high, bool random_sign) {
    std::vector<std::vector<float>> random_vectors;
    for (int i = 0; i < N; ++i) {
        random_vectors.push_back(GenerateRandomVector(dim, low, high, random_sign));
    }
    return random_vectors;
}


HNSW CreateHNSW(int N, int dim, int M, int M0, int ef_construction, float level_multiplier) {
    std::printf("Creating HNSW object, M=%d, M0=%d, ef_construction=%d, m_mult=%f\n",
                M, M0, ef_construction, level_multiplier);
    HNSW hnsw(M, M0, ef_construction, level_multiplier);
    std::printf("HNSW object created\n");

    std::printf("Generating (%d, %d) random vectors...\n", N, dim);
    auto random_vectors = GenerateNRandomVectors(N, dim, 0, 1, true);
    std::printf("Random vectors generated\n");

    std::printf("Inserting vectors to HNSW\n");
    hnsw.InsertBatch(random_vectors);
    std::printf("Vectors processing done\n");

    std::printf("Search process started\n");
    int failed = 0;

    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    high_resolution_clock::time_point end;

    for (int i = 0; i < N; ++i) {
        Points found = hnsw.KNNSearch(random_vectors[i], 10, 10);
        if (found[0] != i) {
            ++failed;
        }
    }

    end = high_resolution_clock::now();
    std::printf("Search process finished, %d/%d failed, %f mean per search\n", failed, N,
                static_cast<double>(duration_cast<microseconds>(end - start).count()) / N / 10e6);

    return hnsw;
}


bool TestHNSWSearch(HNSW &hnsw, const Storage &queries) {
    int failed = 0;

    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    high_resolution_clock::time_point end;

    for (size_t q = 0; q < queries.size(); ++q) {
        Points found = hnsw.KNNSearch(queries[q], 1, 10);
        if (found[0] != static_cast<Point>(q)) {
            ++failed;
        }
    }

    end = high_resolution_clock::now();
    std::printf("Search process finished, %d/%d failed, %f mean per search\n", failed, static_cast<int>(queries.size()),
                static_cast<double>(duration_cast<microseconds>(end - start).count()) / queries.size() / 10e6);
    return failed == 0;
}


template<class T>
bool VectorsEqual(const std::vector<T> &first, const std::vector<T> &second) {
    if (first.size() != second.size()) return false;

    for (size_t i = 0; i < first.size(); ++i) {
        if (std::abs(first[i] - second[i]) > 1e-6) return false;
    }

    return true;
}

template<class T>
void PrintVector(const std::string &prefix, const std::vector<T> &vec) {
    std::cout << prefix << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << ' ';
    }
    std::cout << std::endl;
}


bool TestStorageDump(std::ofstream &ostrm, std::ifstream &istrm, const HNSW &hnsw) {
    std::printf("Testing Storage dump...");
    const Storage &old_storage = hnsw.GetStorage();

    DumpStorage(ostrm, old_storage);
    Storage new_storage = ReadStorageFromDump(istrm);

    bool good = true;
    for (size_t i = 0; i < old_storage.size(); ++i) {
        if (!VectorsEqual(old_storage[i], new_storage[i])) {
            std::printf("\n\tIncorrect vector for Point %d\n", static_cast<int>(i));
            PrintVector("\tOld", old_storage[i]);
            PrintVector("\tNew", new_storage[i]);
            good = false;
            break;
        }
    }
    return good;
}


bool TestLevelsDump(std::ofstream &ostrm, std::ifstream &istrm, const HNSW &hnsw) {
    std::printf("Testing Levels dump...");
    const Levels &old_levels = hnsw.GetLevels();

    DumpLevels(ostrm, old_levels);
    Levels new_levels = ReadLevelsFromDump(istrm);

    bool good = true;
    for (const auto &old_entry : old_levels) {
        if (old_entry.second != new_levels[old_entry.first]) {
            std::printf("\n\tIncorrect level for Point %d\n", static_cast<int>(old_entry.first));
            std::printf("\tOld: %d\n", old_entry.second);
            std::printf("\tNew: %d\n", new_levels[old_entry.first]);
            good = false;
            break;
        }
    }
    return good;
}


template<class T>
std::vector<T> CreateVectorFromSet(const std::unordered_set<T> &set) {
    std::vector<T> new_vec;

    for (const T entry: set) {
        new_vec.push_back(entry);
    }

    std::sort(new_vec.begin(), new_vec.end());

    return new_vec;
}


bool TestHNSWGraphDump(std::ofstream &ostrm, std::ifstream &istrm, const HNSW &hnsw) {
    std::printf("Testing Levels dump...");
    const HNSWGraph &old_graph = hnsw.GetGraph();

    DumpHNSWGraph(ostrm, old_graph);
    HNSWGraph new_graph = ReadHNSWGraphFromDump(istrm);

    bool good = true;
    for (const auto &level : old_graph) {
        int level_index = level.first;

        for (const auto &point_edges : level.second) {
            const Point point = point_edges.first;
            std::vector<Point> old_edges = CreateVectorFromSet(point_edges.second);
            std::vector<Point> new_edges = CreateVectorFromSet(new_graph[level_index][point]);

            if (!VectorsEqual(old_edges, new_edges)) {
                std::printf("\n\tIncorrect edges for Point %d at level %d\n", point_edges.first, level_index);
                PrintVector("\tOld:", old_edges);
                PrintVector("\tNew:", new_edges);
                good = false;
                break;
            }
        }
    }

    return good;
}


bool TestHNSWDump(HNSW &old_hnsw, int K, int ef) {
    std::printf("Testing HNSW dump...");
    const char *storage_file = "test-storage.dump.tmp";
    const char *index_file = "test-index.dump.tmp";
    DumpHNSWToFile(storage_file, index_file, old_hnsw, true);
    HNSW new_hnsw = ReadHNSWFromFile(storage_file, index_file);

    const Storage &queries = new_hnsw.GetStorage();

    bool good = true;
    for (size_t q = 0; q < queries.size(); ++q) {
        auto old_neighbors = old_hnsw.KNNSearch(queries[q], K, ef);
        auto new_neighbors = new_hnsw.KNNSearch(queries[q], K, ef);

        if (!VectorsEqual(old_neighbors, new_neighbors)) {
            std::printf("\n\tIncorrect neighbors for Point %d\n", static_cast<int>(q));
            PrintVector("\tOld:", old_neighbors);
            PrintVector("\tNew:", new_neighbors);
            good = false;
            break;
        }
    }

    std::remove(storage_file);
    std::remove(index_file);
    return good;
}


void RunTests() {
    const char *filename = "test-dump.tmp";
    std::ofstream ostrm(filename, std::ios::binary);
    std::ifstream istrm(filename, std::ios::binary);

    HNSW hnsw = CreateHNSW(100, 3);
    bool test_result;

    test_result = TestStorageDump(ostrm, istrm, hnsw);
    std::printf(test_result ? " ok\n" : " fail\n");
    test_result = TestLevelsDump(ostrm, istrm, hnsw);
    std::printf(test_result ? " ok\n" : " fail\n");
    test_result = TestHNSWGraphDump(ostrm, istrm, hnsw);
    std::printf(test_result ? " ok\n" : " fail\n");
    test_result = TestHNSWDump(hnsw);
    std::printf(test_result ? " ok\n" : " fail\n");

    std::remove(filename);
}
