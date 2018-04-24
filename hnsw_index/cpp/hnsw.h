#ifndef HNSW_HNSW
#define HNSW_HNSW

#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <chrono>

#include "utils.h"
#include "types.h"


class HNSW {
    int max_neighbors{};
    int max_neighbors_0{};
    int ef_construction{};
    float level_multiplier{};
    int max_level = -1;
    Point entry_point = -1;

    Storage storage;
    HNSWGraph graph;
    Levels levels;

public:
    HNSW();

    HNSW(int max_neighbors, int max_neighbors_0, int ef_construction, float level_multiplier);

    HNSW(int max_neighbors, int max_neighbors_0, int ef_construction, float level_multiplier,
         int max_level, Point entry_point, Storage &storage, HNSWGraph &graph, Levels &levels);

    void InsertBatch(Storage batch);

    void Insert(Point new_point);

    Points KNNSearch(const Coords &query, int K, int ef);

    const Storage& GetStorage() const;

    const Levels& GetLevels() const;

    const HNSWGraph& GetGraph() const;

    const int GetMaxLevel() const;

    const Point GetEntryPoint() const;

    const int GetMaxNeighbors() const;

    const int GetMaxNeighbors0() const;

    const int GetEfConstruction() const;

    const float GetLevelMultiplier() const;

private:
    void TrimNeighbors(Point element_id, int max_neighbors, int level);

    void MutuallyConnect(Point first, Point second, int level);

    PointsSet SelectBestNeighbors(LessDistanceQueue &candidates, Point point, int max_neighbors, int level,
                                  bool extend_candidates=false, bool keep_pruned=false);

    LessDistanceQueue SearchLevel(const Coords &query, PointsSet &entry_points_set, int max_neighbors, int level);

    int GenerateLevel();

    Coords& GetCoords(Point query);
};

#endif // HNSW_HNSW