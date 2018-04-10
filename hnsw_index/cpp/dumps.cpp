#include <fstream>
#include "hnsw.h"
#include "dumps.h"
#include "types.h"


template<class Iterable>
void DumpIterable(std::ofstream &ofstream, const Iterable &iter) {
    size_t iter_size = iter.size();

    ofstream << iter_size << ' ';
    for (const auto i : iter) {
        ofstream << i << ' ';
    }

    ofstream << '\n';
    ofstream.flush();
}

template<class T>
std::vector<T> ReadVectorFromDump(std::ifstream &ifstream) {
    size_t new_vec_size;
    ifstream >> new_vec_size;

    std::vector<T> new_vec(new_vec_size);
    for (size_t i = 0; i < new_vec_size; ++i) {
        T v;
        ifstream >> v;
        new_vec[i] = v;
    }
    return new_vec;
}


void DumpStorage(std::ofstream &ofstream, const Storage &storage) {
    size_t storage_size = storage.size();
    ofstream << storage_size << '\n';

    for (const Coords &point_coords : storage) {
        DumpIterable(ofstream, point_coords);
    }
    ofstream.flush();
}


Storage ReadStorageFromDump(std::ifstream &ifstream) {
    size_t storage_size;
    ifstream >> storage_size;

    Storage new_storage(storage_size);
    for (size_t i = 0; i < storage_size; ++i) {
        new_storage[i] = ReadVectorFromDump<float>(ifstream);
    }
    return new_storage;
}


void DumpLevels(std::ofstream &ofstream, const Levels &levels) {
    size_t levels_size = levels.size();
    ofstream << levels_size << '\n';

    for (const auto &entry : levels) {
        ofstream << entry.first << ' ' << entry.second << '\n';
    }
    ofstream.flush();
}


Levels ReadLevelsFromDump(std::ifstream &ifstream) {
    size_t levels_size;
    ifstream >> levels_size;

    Levels new_levels;
    for (size_t i = 0; i < levels_size; ++i) {
        Point p;
        int level;
        ifstream >> p >> level;
        new_levels[p] = level;
    }
    return new_levels;
}


void DumpHNSWGraph(std::ofstream &ofstream, const HNSWGraph &graph) {
    size_t levels_num = graph.size();
    ofstream << levels_num << '\n';

    for (const auto &level : graph) {
        int level_index = level.first;
        ofstream << level_index << ' ' << level.second.size() << '\n';

        for (const auto &point_edges : level.second) {
            ofstream << point_edges.first << '\n';
            DumpIterable(ofstream, point_edges.second);
        }
    }

    ofstream.flush();
}


HNSWGraph ReadHNSWGraphFromDump(std::ifstream &ifstream) {
    size_t levels_num;
    ifstream >> levels_num;

    HNSWGraph new_graph;
    for (size_t i = 0; i < levels_num; ++i) {
        int cur_level;
        size_t level_size;
        ifstream >> cur_level >> level_size;

        for (size_t j = 0; j < level_size; ++j) {
            Point point;
            ifstream >> point;
            for (Point p : ReadVectorFromDump<Point>(ifstream)) {
                new_graph[cur_level][point].insert(p);
            }
        }
    }
    return new_graph;
}


void DumpHNSWToFile(const std::string &storage_file, const std::string &index_file,
                    const HNSW &hnsw, bool dump_storage) {
    if (dump_storage) {
        std::ofstream storage_ostrm(storage_file, std::ios::binary);
        DumpStorage(storage_ostrm, hnsw.GetStorage());
    }

    std::ofstream index_ostrm(index_file, std::ios::binary);
    index_ostrm << hnsw.GetMaxLevel() << ' ' << hnsw.GetEntryPoint() << '\n';
    index_ostrm << hnsw.GetMaxNeighbors() << ' ' << hnsw.GetMaxNeighbors0() << '\n';
    index_ostrm << hnsw.GetEfConstruction() << ' ' << hnsw.GetLevelMultiplier() << '\n';
    DumpHNSWGraph(index_ostrm, hnsw.GetGraph());
    DumpLevels(index_ostrm, hnsw.GetLevels());
}


HNSW ReadHNSWFromFile(const std::string &storage_file, const std::string &index_file) {
    std::ifstream storage_istrm(storage_file, std::ios::binary);
    Storage storage = ReadStorageFromDump(storage_istrm);
    std::ifstream index_istrm(index_file, std::ios::binary);

    int max_level, max_neighbors, max_neighbors_0, ef_construction;
    float level_multiplier;
    Point entry_point;
    index_istrm >> max_level >> entry_point;
    index_istrm >> max_neighbors >> max_neighbors_0;
    index_istrm >> ef_construction >> level_multiplier;

    HNSWGraph graph = ReadHNSWGraphFromDump(index_istrm);
    Levels levels = ReadLevelsFromDump(index_istrm);

    return HNSW(max_neighbors, max_neighbors_0, ef_construction, level_multiplier,
                max_level, entry_point, storage, graph, levels);
}
