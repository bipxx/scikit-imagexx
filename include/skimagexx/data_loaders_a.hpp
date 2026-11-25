#pragma once

#include <cstddef>

namespace skimagexx {

// Copy the decoded bytes of a sample dataset into fresh caller-owned storage.
// Both buffers must cover their stated extents and must not overlap.
void copy_sample_data_into(const void* source, std::size_t source_bytes,
                           void* destination, std::size_t destination_bytes);

}  // namespace skimagexx
