#include <utility>
#include <vector>
#include <cmath>
#include <chrono>

#include "utils.h"
#include "hnsw.h"


HNSW::HNSW() = default;


HNSW::HNSW(int max_neighbors, int max_neighbors_0, int ef_construction, float level_multiplier) :
    max_neighbors(max_neighbors),
    max_neighbors_0(max_neighbors_0),
    ef_construction(ef_construction),
    level_multiplier(level_multiplier) {}

HNSW::HNSW(int max_neighbors, int max_neighbors_0, int ef_construction, float level_multiplier,
           int max_level, Point entry_point, Storage &storage, HNSWGraph &graph, Levels &levels) :
    max_neighbors(max_neighbors),
    max_neighbors_0(max_neighbors_0),
    ef_construction(ef_construction),
    level_multiplier(level_multiplier),
    max_level(max_level),
    entry_point(entry_point),
    storage(storage),
    graph(graph),
    levels(levels) {}

void HNSW::InsertBatch(Storage batch) {
    int log_step = 100;
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    high_resolution_clock::time_point end;

    for (const Coords &coords : batch) {
        auto new_point = static_cast<Point>(storage.size());
        if (new_point % log_step == 0) {
            end = high_resolution_clock::now();
            std::printf("\t%d %f per point\n", new_point,
                        static_cast<double>(duration_cast<microseconds>(end - start).count()) / log_step / 10e6);
            start = end;
        }
        storage.push_back(coords);
        Insert(new_point);
    }
}

void HNSW::Insert(Point new_point) {
    int level = GenerateLevel();
    levels[new_point] = level;

    PointsSet entry_points_set = entry_point < 0 ? PointsSet() : PointsSet{entry_point};

    for (int cur_level = max_level; cur_level > level; --cur_level) {
        LessDistanceQueue best_candidates = SearchLevel(new_point, entry_points_set, 1, cur_level);
        entry_points_set = {best_candidates.top().id};
    }

    int start_level = std::min(max_level, level);
    for (int cur_level = start_level; cur_level >= 0; --cur_level) {
        int M = cur_level > 0 ? max_neighbors : max_neighbors_0;
        LessDistanceQueue best_candidates = SearchLevel(new_point, entry_points_set, ef_construction, cur_level);
        entry_points_set = SelectBestNeighbors(best_candidates, new_point, M, cur_level);

        for (Point neighbor : entry_points_set) {
            MutuallyConnect(new_point, neighbor, cur_level);
            TrimNeighbors(neighbor, M, cur_level);
        }
    }

    if (level > max_level) {
        max_level = level;
        entry_point = new_point;
    }
}

Points HNSW::KNNSearch(const Coords &query, int K, int ef) {
    query_coords = query;
    PointsSet entry_points_set{entry_point};

    for (int cur_level = max_level; cur_level > 0; --cur_level) {
        LessDistanceQueue best_candidates = SearchLevel(query_id, entry_points_set, 1, cur_level);
        entry_points_set = {best_candidates.top().id};
    }

    LessDistanceQueue best_candidates = SearchLevel(query_id, entry_points_set, ef, 0);

    Points points;
    while (points.size() < static_cast<size_t>(K) and !best_candidates.empty()) {
        points.push_back(best_candidates.top().id);
        best_candidates.pop();
    }

    return points;
}

const Storage& HNSW::GetStorage() const {
    return storage;
}

const Levels& HNSW::GetLevels() const {
    return levels;
}

const HNSWGraph& HNSW::GetGraph() const {
    return graph;
}

const int HNSW::GetMaxLevel() const {
    return max_level;
}

const Point HNSW::GetEntryPoint() const {
    return entry_point;
}

const int HNSW::GetMaxNeighbors() const {
    return max_neighbors;
}

const int HNSW::GetMaxNeighbors0() const {
    return max_neighbors_0;
}

const int HNSW::GetEfConstruction() const {
    return ef_construction;
}

const float HNSW::GetLevelMultiplier() const {
    return level_multiplier;
}

void HNSW::TrimNeighbors(Point element_id, int max_neighbors, int level) {
    PointsSet neighbors = graph[level][element_id];

    if (neighbors.size() > static_cast<size_t>(max_neighbors)) {
        std::vector<Distance> distances;
        for (Point n: neighbors) {
            distances.emplace_back(n, GetCoords(element_id), GetCoords(n));
        }

        LessDistanceQueue candidates(distances);
        graph[level][element_id] = SelectBestNeighbors(candidates, element_id, max_neighbors, level);
    }
}

void HNSW::MutuallyConnect(Point first, Point second, int level) {
    graph[level][first].insert(second);
    graph[level][second].insert(first);
}

PointsSet HNSW::SelectBestNeighbors(LessDistanceQueue &candidates, Point point, int max_neighbors, int level,
                              bool extend_candidates, bool keep_pruned) {
    PointsSet best_neighbors;
    LessDistanceQueue pruned_neighbors;

    if (extend_candidates) {
        PointsSet extended_candidates;
        LessDistanceQueue tmp = candidates;

        while (!tmp.empty()) {
            for (Point p: graph[level][tmp.top().id]) {
                extended_candidates.insert(p);
            }
            tmp.pop();
        }

        for (Point p: extended_candidates) {
            candidates.push(Distance(p, GetCoords(point), GetCoords(p)));
        }
    }

    while (!candidates.empty()) {
        if (best_neighbors.size() >= static_cast<size_t>(max_neighbors)) break;

        Distance cand_q = candidates.top();
        candidates.pop();

        // distance between query and candidate should be shortest candidate edge (NSW)
        bool good = true;
        for (Point n: best_neighbors) {
            Distance cand_n = Distance(n, GetCoords(n), GetCoords(cand_q.id));

            if (cand_n.dist < cand_q.dist) {
                good = false;
                pruned_neighbors.push(cand_q);
                break;
            }
        }

        if (good) {
            best_neighbors.insert(cand_q.id);
        }
    }

    if (keep_pruned && best_neighbors.size() < static_cast<size_t>(max_neighbors)) {
        while (!pruned_neighbors.empty() && best_neighbors.size() < static_cast<size_t>(max_neighbors)) {
            best_neighbors.insert(pruned_neighbors.top().id);
            pruned_neighbors.pop();
        }
    }

    return best_neighbors;
}

LessDistanceQueue HNSW::SearchLevel(Point point, PointsSet &entry_points_set, int max_neighbors, int level) {
    std::vector<Distance> distances;
    for (Point n: entry_points_set) {
        distances.emplace_back(n, GetCoords(point), GetCoords(n));
    }

    LessDistanceQueue candidates(distances);
    MoreDistanceQueue neighbors(distances);
    PointsSet visited(entry_points_set);

    while (!candidates.empty()) {
        Distance candidate = candidates.top();
        candidates.pop();

        if (candidate.dist > neighbors.top().dist) break;

        for (Point e: graph[level][candidate.id]) {
            if (visited.find(e) == visited.end()) {
                visited.insert(e);

                Distance e_dist(e, GetCoords(point), GetCoords(e));
                if (e_dist.dist < neighbors.top().dist || neighbors.size() < static_cast<size_t>(max_neighbors)) {
                    neighbors.push(e_dist);
                    candidates.push(e_dist);

                    if (neighbors.size() > static_cast<size_t>(max_neighbors)) {
                        neighbors.pop();
                    }
                }
            }
        }
    }

    LessDistanceQueue neighbors_selected;
    while (!neighbors.empty()) {
        neighbors_selected.push(neighbors.top());
        neighbors.pop();
    }

    return neighbors_selected;
}

int HNSW::GenerateLevel() {
    float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    return static_cast<int>(std::floor(-std::log(r) * level_multiplier));
}

Coords& HNSW::GetCoords(const Point query) {
    if (query == query_id) {
        return query_coords;
    } else {
        return storage[query];
    }
}
