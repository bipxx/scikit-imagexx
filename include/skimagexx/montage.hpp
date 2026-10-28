#pragma once
#include <cstddef>

namespace skimagexx::detail {
struct MontageInput {
    const void* data;
    std::size_t shape[4]; // images, rows, columns, channels
    std::ptrdiff_t strides[4]; // signed BYTE strides
    std::size_t itemsize;
};
struct MontageGrid { std::size_t rows,columns,padding; };
struct MontageOutput { void* data; std::size_t rows,columns; };
// Boundary invariants: stable prepared POD words, itemsize 1/2/4/8/16; every
// signed input span and product is addressable. Typed fill holds channels
// consecutive same-dtype words. Output is fresh/disjoint contiguous HWC storage
// with the input dtype, and is disjoint from input/fill/descriptors. Grid has
// capacity for every image; output dimensions equal (tile_extent+padding)*grid
// extent+padding. Positive image count requires positive grid columns. Empty
// axes are supported; null storage is permitted only when no word is accessed.
// Caller snapshots workers >= 1 once. No validation, allocation, conversion,
// floating arithmetic, Python calls or global worker mutation occurs here.
// Fill precedes independent tile-row copies; image index follows grid C order.
// No refcounted/object or padded/undefined-representation words are admitted.
void montage_into(const MontageInput&,MontageGrid,const void* typed_channel_fill,
                  MontageOutput,int workers);
}
