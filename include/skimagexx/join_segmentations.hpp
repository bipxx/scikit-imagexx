#pragma once
#include <cstddef>
#include <cstdint>
#include "components.hpp"
#include <vector>

namespace skimagexx {
// NumPy resolves scalar/array promotion before these pixel stages. Integer
// operations wrap in each declared dtype, including the intermediate product.
// Buffers are native-endian, contiguous C order, with count elements and no
// output/input overlap. Integer input labels and divmod input are nonnegative.
// factor is positive; divmod's dtype is the promoted integer result dtype.
// These stages do not implement ndarray/provider metadata or warning callbacks.
struct JoinType { unsigned bytes; bool is_signed; bool floating=false; };
void join_encode(const void* left, JoinType left_type,
                 const void* right, JoinType right_type, std::size_t count,
                 std::uint64_t factor, JoinType product_type, JoinType sum_type,
                 void* output);
void join_divmod(const void* labels, JoinType label_type, std::size_t count,
                 std::uint64_t factor, JoinType result_type,
                 void* quotient, void* remainder);

// A standalone caller declares its scalar/array promotion choices explicitly.
// NumPy1 value-based and NumPy2 strong scalar promotion are different policies;
// this SDK boundary intentionally does not invent a replacement type system.
struct JoinParameters {
    JoinType product_type, sum_type, division_type;
    bool return_mapping=false;
};
struct JoinArray {
    JoinType type;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> data;
};
struct JoinMapView {const JoinArray* keys;const JoinArray* values;};
struct JoinResult {
    JoinArray joined,keys,first_values,second_values;
    bool has_mapping=false;
    JoinMapView first_map() const {return {&keys,&first_values};}
    JoinMapView second_map() const {return {&keys,&second_values};}
};
// Source views use the existing ComponentsView bounded-storage contract and
// logical C traversal, including noncontiguous/reversed native integer views.
// The operation computes the ORIGINAL right maximum+uint8(1), relabels both
// fields, wraps multiply/add separately, and returns owned typed C-order data.
// Nonpositive factor and floating-promotion plans are explicit SDK frontiers.
JoinResult join_segmentations(const ComponentsView& left,const ComponentsView& right,
                              const JoinParameters& parameters);
}
