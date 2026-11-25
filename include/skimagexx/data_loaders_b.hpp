#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {
enum class DataRowOrder { top_down, bottom_up };

// Copy tightly packed, non-overlapping image buffers into top-down order.
// Both lengths must equal rows * row_bytes.
void copy_data_rows(const std::uint8_t* source, std::size_t source_size,
                    std::uint8_t* destination, std::size_t destination_size,
                    std::size_t rows, std::size_t row_bytes,
                    DataRowOrder order = DataRowOrder::top_down);
} // namespace skimagexx
