#include <getopt.h>
#include <iostream>
#include "hnsw.h"
#include "dumps.h"
#include "tests.h"


bool build, load, test;
int max_neighbors, max_neighbors_0, ef_construction;
std::string storage_path, params_path;
float level_multiplier;


void PrintHelp() {
    std::cout <<
        "--build (-b)                    Build index and write it to storage & params\n"
        "--load (-l)                     Load index params from storage & params\n"
        "--test (-t)                     Test index after build\n"
        "--max-neighbors (-N) <int>:     Degree limit for level > 0\n"
        "--max-neighbors-0 (-n) <int>:   Degree limit for level 0\n"
        "--ef-construction (-e) <int>:   Degree limit during build\n"
        "--level-mult (-m) <float>:      Level multiplier during build\n"
        "--storage (-s) <fname>:         File to read/write storage\n"
        "--params (-p) <fname>:          File to read/write params\n"
        "--help (-h):                    Show help\n";
    exit(1);
}


void ProcessArgs(int argc, char** argv) {
    const char* const short_opts = "bltN:n:e:m:s:p:h";
    const option long_opts[] = {
            // const char *name; int has_arg; int *flag; int val;
            // int has_arg: [0 - no arg, 1 - required, 2 - not required];
            // int *flag: if nullptr: val, else: 0 & *flag = val;
            // val = optstring;
            {"build", 0, nullptr, 'b'},
            {"load", 0, nullptr, 'l'},
            {"test", 0, nullptr, 't'},

            {"max_neighbors", 1, nullptr, 'N'},
            {"max_neighbors_0", 1, nullptr, 'n'},
            {"ef_construction", 1, nullptr, 'e'},
            {"level_multiplier", 1, nullptr, 'm'},

            {"storage_path", 1, nullptr, 's'},
            {"params_path", 1, nullptr, 'p'},

            {"help", 0, nullptr, 'h'},
            {nullptr, 0, nullptr, 0},  // last element must be zeros
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt) break;
        switch (opt) {
            case 'b':
                build = true;
                std::cout << "build is set to true\n";
                break;

            case 'l':
                load = true;
                std::cout << "load is set to true\n";
                break;

            case 't':
                test = true;
                std::cout << "test is set to true\n";
                break;

            case 'N':
                max_neighbors = std::stoi(optarg);
                std::cout << "max_neighbors is set to " << max_neighbors << std::endl;
                break;

            case 'n':
                max_neighbors_0 = std::stoi(optarg);
                std::cout << "max_neighbors_0 is set to " << max_neighbors_0 << std::endl;
                break;

            case 'e':
                ef_construction = std::stoi(optarg);
                std::cout << "ef_construction is set to " << ef_construction << std::endl;
                break;

            case 'm':
                level_multiplier = std::stof(optarg);
                std::cout << "level_multiplier is set to " << level_multiplier << std::endl;
                break;

            case 's':
                storage_path = std::string(optarg);
                std::cout << "storage_path file set to: " << storage_path << std::endl;
                break;

            case 'p':
                params_path = std::string(optarg);
                std::cout << "params_path file set to: " << params_path << std::endl;
                break;

            case 'h': // -h or --help
            case '?': // Unrecognized option
            default:
                PrintHelp();
                break;
        }
    }
}


void ValidateArgs() {
    if (storage_path.empty() || params_path.empty()) {
        std::cout << "--storage (for load) and --params (for dump/load) must be set" << std::endl;
        exit(1);
    }

    if ((build && load) || !(load || build)) {
        std::cout << "one of --build or --load must be set" << std::endl;
        exit(1);
    }

    if (build && !(max_neighbors && max_neighbors_0 && ef_construction && level_multiplier)) {
        std::cout << "--max-neighbors, --max-neighbors-0, --ef-construction, --level-mult must be set" << std::endl;
        exit(1);
    }
}


int main(int argc, char **argv) {
    ProcessArgs(argc, argv);
    ValidateArgs();

    HNSW hnsw;
    if (build) {
        std::cout << "Loading data from " << storage_path << "...\n";
        std::ifstream istream(storage_path, std::ios::binary);
        Storage storage = ReadStorageFromDump(istream);

        std::cout << "Building index...\n";
        hnsw = HNSW(max_neighbors, max_neighbors_0, ef_construction, level_multiplier);
        hnsw.InsertBatch(storage);

        std::cout << "Writing index params to " << params_path << "... \n";
        DumpHNSWToFile(storage_path, params_path, hnsw, false);

    } else {
        std::cout << "Loading storage from "
                  << storage_path
                  << " and index params from "
                  << params_path << "...\n";

        hnsw = ReadHNSWFromFile(storage_path, params_path);
    }

    if (test) {
        std::cout << "Testing index...\n";
        TestHNSWSearch(hnsw, hnsw.GetStorage());
    }

    return 0;
}
