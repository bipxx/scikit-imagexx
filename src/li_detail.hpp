#pragma once
#include <cstddef>
#include <limits>
namespace skimagexx::detail {
struct LiScan {
    std::size_t valid=0,finite=0;
    double low=std::numeric_limits<double>::infinity(),high=-std::numeric_limits<double>::infinity();
    bool constant=true;
};
}
