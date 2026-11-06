#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace label_pair_fixture {
struct Original {const char* id;std::vector<std::int64_t> truth,test,ignored;bool normalize;std::size_t rows,columns;std::vector<std::int64_t> indptr,indices;std::vector<std::uint64_t> data_bits;unsigned status;};
inline const std::array<Original,5> originals={{
{"contingency_table/duplicates",{0,1,1,2,2,2},{0,2,2,1,1,2},{},false,3,3,{0,1,2,4},{0,2,1,2},{UINT64_C(0x3ff0000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x3ff0000000000000)},0},
{"contingency_table/normalize-ignore",{0,1,1,2,2,2},{0,2,2,1,1,2},{0},true,3,3,{0,1,2,4},{0,2,1,2},{UINT64_C(0x0000000000000000),UINT64_C(0x3fd999999999999a),UINT64_C(0x3fd999999999999a),UINT64_C(0x3fc999999999999a)},0},
{"contingency_table/allignored-normalize",{1,1,1,1,1,1},{0,2,2,1,1,2},{1},true,2,3,{0,0,3},{0,1,2},{UINT64_C(0xfff8000000000000),UINT64_C(0xfff8000000000000),UINT64_C(0xfff8000000000000)},1},
{"contingency_table/unequal-shapes-flat-match",{0,1,1,2,2,2},{0,2,2,1,1,2},{},false,3,3,{0,1,2,4},{0,2,1,2},{UINT64_C(0x3ff0000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x3ff0000000000000)},0},
{"contingency_table/readonly-reversed",{1,1,0,2,2,2},{2,2,0,2,1,1},{},false,3,3,{0,1,2,4},{0,2,1,2},{UINT64_C(0x3ff0000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x3ff0000000000000)},0},
}};
}
