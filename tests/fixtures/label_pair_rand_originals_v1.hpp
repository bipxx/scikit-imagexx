#pragma once
#include <array>
#include <cstdint>
#include <vector>
namespace label_pair_rand_originals {
struct Original {const char* id;std::vector<std::int64_t> truth,test,ignored;double alpha;std::array<double,3> terms;std::array<std::uint64_t,3> words;};
inline const std::vector<Original> rows {
{"adapted_rand_error/perfect",{0,1,1,2,2,2},{0,1,1,2,2,2},{0},0.5,{8.,8.,8.},{UINT64_C(0x0000000000000000),UINT64_C(0x3ff0000000000000),UINT64_C(0x3ff0000000000000)}},
{"adapted_rand_error/split-alpha0",{1,1,1,1,1,1},{1,2,3,4,5,6},{},0.0,{0.,30.,0.},{UINT64_C(0xfff8000000000000),UINT64_C(0x0000000000000000),UINT64_C(0xfff8000000000000)}},
{"adapted_rand_error/merge-alpha1",{1,2,3,4,5,6},{1,1,1,1,1,1},{},1.0,{0.,0.,30.},{UINT64_C(0xfff8000000000000),UINT64_C(0xfff8000000000000),UINT64_C(0x0000000000000000)}},
{"adapted_rand_error/quarter-alpha",{0,1,1,2,2,2},{0,2,2,1,1,2},{0},0.25,{4.,8.,8.},{UINT64_C(0x3fe0000000000000),UINT64_C(0x3fe0000000000000),UINT64_C(0x3fe0000000000000)}},
{"adapted_rand_error/singleton",{1},{1},{0},0.5,{0.,0.,0.},{UINT64_C(0xfff8000000000000),UINT64_C(0xfff8000000000000),UINT64_C(0xfff8000000000000)}},
};
}
