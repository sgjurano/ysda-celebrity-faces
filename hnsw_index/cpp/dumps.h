#ifndef HNSW_DUMPS
#define HNSW_DUMPS

#include <fstream>
#include <vector>
#include <iostream>
#include "hnsw.h"


template<class Iterable>
void DumpIterable(std::ofstream &ofstream, const Iterable &iter);


template<class T>
std::vector<T> ReadVectorFromDump(std::ifstream &ifstream);


void DumpStorage(std::ofstream &ofstream, const Storage &storage);


Storage ReadStorageFromDump(std::ifstream &ifstream);


void DumpLevels(std::ofstream &ofstream, const Levels &levels);


Levels ReadLevelsFromDump(std::ifstream &ifstream);


void DumpHNSWGraph(std::ofstream &ofstream, const HNSWGraph &graph);


HNSWGraph ReadHNSWGraphFromDump(std::ifstream &ifstream);


void DumpHNSWToFile(const std::string &storage_file, const std::string &index_file,
                    const HNSW &hnsw, bool dump_storage=false);


HNSW ReadHNSWFromFile(const std::string &storage_file, const std::string &index_file);

#endif // HNSW_DUMPS