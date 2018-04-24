#include <functional>
#include <queue>
#include <cmath>
#include <unordered_set>
#include "utils.h"


Distance::Distance(int id, const Coords &target, const Coords &element) : id(id), dist(0) {
    dist = ComputeDistance(target, element);
};

double Distance::ComputeDistance(const Coords &first, const Coords &second) {
    double dist = 0;
    for (size_t i = 0; i < first.size(); ++i) {
        dist += std::pow(first[i] - second[i], 2);
    }
    return std::sqrt(dist / first.size());
}

bool Distance::operator<(const Distance &other) const {
    return dist < other.dist;
}

bool Distance::operator>(const Distance &other) const {
    return dist > other.dist;
}


LessDistanceQueue::LessDistanceQueue(std::vector<Distance> &vector) {
    for (Distance i : vector) {
        queue.push(i);
    }
}

LessDistanceQueue::LessDistanceQueue() = default;

void LessDistanceQueue::push(Distance vertex) {
    queue.push(vertex);
}

void LessDistanceQueue::pop() {
    return queue.pop();
}

Distance const LessDistanceQueue::top() {
    return queue.top();
}

bool LessDistanceQueue::empty() {
    return queue.empty();
}

size_t LessDistanceQueue::size() {
    return queue.size();
}


MoreDistanceQueue::MoreDistanceQueue(std::vector<Distance> &vector) {
    for (Distance i : vector) {
        queue.push(i);
    }
}

MoreDistanceQueue::MoreDistanceQueue() = default;

void MoreDistanceQueue::push(Distance vertex) {
    queue.push(vertex);
}

void MoreDistanceQueue::pop() {
    return queue.pop();
}

Distance const MoreDistanceQueue::top() {
    return queue.top();
}

bool MoreDistanceQueue::empty() {
    return queue.empty();
}

size_t MoreDistanceQueue::size() {
    return queue.size();
}
