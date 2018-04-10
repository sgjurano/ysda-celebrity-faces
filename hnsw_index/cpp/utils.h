#ifndef HNSW_UTILS
#define HNSW_UTILS

#include <functional>
#include <queue>
#include <cmath>
#include "types.h"


struct Distance {
    int id;
    double dist;

    Distance(int id, Coords &target, Coords &element);

    double ComputeDistance(Coords &first, Coords &second);

    bool operator<(const Distance &other) const;

    bool operator>(const Distance &other) const;
};


class LessDistanceQueue {
    std::priority_queue<Distance, std::vector<Distance>, std::greater<>> queue;

public:
    explicit LessDistanceQueue(std::vector<Distance> &vector);

    LessDistanceQueue();

    void push(Distance vertex);

    void pop();

    Distance const top();

    bool empty();

    size_t size();
};


class MoreDistanceQueue {
    std::priority_queue<Distance, std::vector<Distance>, std::less<>> queue;

public:
    explicit MoreDistanceQueue(std::vector<Distance> &vector);

    MoreDistanceQueue();

    void push(Distance vertex);

    void pop();

    Distance const top();

    bool empty();

    size_t size();
};

#endif // HNSW_UTILS