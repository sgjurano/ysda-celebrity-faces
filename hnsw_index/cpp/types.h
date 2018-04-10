#ifndef HNSW_TYPES_H
#define HNSW_TYPES_H

#include <vector>
#include <unordered_set>
#include <unordered_map>

typedef int Point;
typedef std::unordered_set<Point> PointsSet;
typedef std::vector<Point> Points;

typedef std::vector<float> Coords;
typedef std::vector<Coords> Storage;
typedef std::unordered_map<int, std::unordered_map<Point, PointsSet>> HNSWGraph;
typedef std::unordered_map<Point, int> Levels;

#endif //HNSW_TYPES_H
