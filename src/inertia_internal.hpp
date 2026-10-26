#pragma once
#include <cstddef>
namespace skimagexx::detail {
// Caller supplies aligned native buffers: diagonal[n], mixed[n*(n-1)/2]
// in lexicographic pair order, and disjoint output[n*n]. Sum is the source
// NumPy sum, not a reordered sum excluding the current diagonal. No workers.
void inertia_tensor(const float*,const float*,std::size_t,float,float,float*);
void inertia_tensor(const double*,const double*,std::size_t,double,double,double*);
}
