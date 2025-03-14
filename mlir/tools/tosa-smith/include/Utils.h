#ifndef TOSA_SMITH_UTILS_H
#define TOSA_SMITH_UTILS_H

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/MLIRContext.h"
#include <random>
#include <algorithm>

namespace tosas {

static std::random_device rd;
static std::mt19937 rng(rd());

inline unsigned random(int max, int start = 0) {
    assert(start <= max
        && "The start of the random range should be less or equal to the end!");

    if (start == max)
        return start;

    std::uniform_int_distribution<> distrib(start, max);
    return distrib(rng);
}

template<typename T>
T random(const llvm::DenseSet<T> &container) {
    auto iter = container.begin();
    if (container.size() == 1)
        return *iter;

    unsigned itemIndex = random(container.size() - 1);
    std::advance(iter, itemIndex);

    return *iter;
}

template<typename T>
T random(const llvm::SmallVector<T> &container) {
    auto iter = container.begin();
    if (container.size() == 1)
        return *iter;

    unsigned itemIndex = random(container.size() - 1);
    std::advance(iter, itemIndex);

    return *iter;
}

inline int64_t randomInteger() {
    return random(9999, -9999);
}

double randomRadian() {
    unsigned degree = random(360, 0);
    assert(0 <= degree && degree <= 360 && "The range of a degree is [0, 360]");
    double pi = 3.14159265359;
    return ((double) degree) * pi / 180.0;
}

} // namespace tosas


#endif