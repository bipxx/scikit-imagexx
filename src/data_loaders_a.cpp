#include "skimagexx/data_loaders_a.hpp"

#include <cstring>
#include <stdexcept>

namespace skimagexx {

void copy_sample_data_into(const void* source, std::size_t source_bytes,
                           void* destination, std::size_t destination_bytes) {
    if (source_bytes != destination_bytes ||
        (source_bytes != 0 && (source == nullptr || destination == nullptr))) {
        throw std::invalid_argument("sample data buffers have incompatible extents");
    }
    if (source_bytes != 0) {
        std::memcpy(destination, source, source_bytes);
    }
}

}  // namespace skimagexx
