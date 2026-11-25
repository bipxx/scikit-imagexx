#include "skimagexx/data_loaders_b.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace skimagexx {
void copy_data_rows(const std::uint8_t* source, std::size_t source_size,
                    std::uint8_t* destination, std::size_t destination_size,
                    std::size_t rows, std::size_t row_bytes, DataRowOrder order) {
    if (!source || !destination || !rows || !row_bytes ||
        rows > std::numeric_limits<std::size_t>::max() / row_bytes ||
        source_size != rows * row_bytes || destination_size != source_size) {
        throw std::invalid_argument("invalid packed image buffer");
    }
    if (order == DataRowOrder::top_down) {
        std::memcpy(destination, source, source_size);
        return;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        std::memcpy(destination + row * row_bytes,
                    source + (rows - 1 - row) * row_bytes, row_bytes);
    }
}
} // namespace skimagexx
