#pragma once
#include <cstddef>
namespace skimagexx::detail {
// Positive unique counts in sorted-symbol order; empty is allowed.
// The sum of counts is the number of addressable input elements.
double shannon_entropy_counts(const std::ptrdiff_t* counts,std::size_t size,int workers);
}
