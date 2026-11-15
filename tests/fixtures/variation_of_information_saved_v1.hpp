#pragma once
// Exact retained source recipes and RAW saved original words, never computed gold.
#include <skimagexx/variation_of_information.hpp>
#include <array>
#include <vector>
struct ViSaved {const char* id;std::vector<std::int64_t> truth,test,ignored;bool supplied;std::vector<double> data;std::array<std::uint64_t,2> words;};
inline const std::vector<ViSaved> vi_saved={
{"variation_of_information/perfect",{0,1,1,2,2,2},{0,1,1,2,2,2},{},false,{},{UINT64_C(0x0000000000000000),UINT64_C(0x0000000000000000)}},
{"variation_of_information/split",{1,1,1,1,1,1},{1,2,3,4,5,6},{},false,{},{UINT64_C(0x4004ae00d1cfdeb4),UINT64_C(0x0000000000000000)}},
{"variation_of_information/merge-F",{1,2,3,4,5,6},{1,1,1,1,1,1},{},false,{},{UINT64_C(0x0000000000000000),UINT64_C(0x4004ae00d1cfdeb4)}},
{"variation_of_information/ignore",{0,1,1,2,2,2},{0,2,2,1,1,2},{0},false,{},{UINT64_C(0x3fe1a19b9126167e),UINT64_C(0x3fe1a19b9126167e)}},
{"variation_of_information/allignored",{1,1,1,1,1,1},{0,2,2,1,1,2},{1},false,{},{UINT64_C(0xfff8000000000000),UINT64_C(0xfff8000000000000)}},
{"variation_of_information/table-normalized",{0,1,1,2,2,2},{0,2,2,1,1,2},{},true,{0.125,0.25,0.625,0.0},{UINT64_C(0x3fd60a02756f9c1d),UINT64_C(0x3fdf3379bb7f0704)}},
{"variation_of_information/table-raw-counts",{0,1,1,2,2,2},{0,2,2,1,1,2},{},true,{1.0,2.0,3.0,0.0},{UINT64_C(0x40060a02756f9c1d),UINT64_C(0x4009f5fd8a9063e3)}},
};
