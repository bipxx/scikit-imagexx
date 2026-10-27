#include "polygon.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using skimagexx::detail::PolygonView;
using skimagexx::detail::points_in_poly;
using skimagexx::detail::grid_points_in_poly;
namespace {
std::size_t controls=0;std::ofstream journal;
void check(bool yes){++controls;assert(yes);}
struct Original {
    const char* id;bool grid;std::size_t rows,columns;
    std::vector<double> vx,vy,x,y;std::vector<std::uint8_t> expected;
};
// Literal values decoded from saved pre-candidate original observations.
const std::vector<Original> originals={
{"points_in_poly-square",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"points_in_poly-concave",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{2,0,2,3,2,3,0,3}},
{"points_in_poly-closed",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"points_in_poly-clockwise",false,0,0,{0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0,0x0.0p+0},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"points_in_poly-bowtie",false,0,0,{0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{0,0,2,0,2,0,3,3}},
{"points_in_poly-degenerate-one",false,0,0,{0x0.0p+0},{0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{0,0,2,0,0,0,0,0}},
{"points_in_poly-degenerate-two",false,0,0,{0x0.0p+0,0x0.0p+0},{0x0.0p+0,0x1.0000000000000p+2},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{0,0,2,0,0,0,0,0}},
{"points_in_poly-epsilon",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x0.0p+0,0x1.19799812dea11p-41,0x1.19799812dea11p-40,0x1.19799812dea10p-40,-0x1.19799812dea11p-41,-0x1.19799812dea11p-40,0x1.0000000000000p+1},{0x0.0p+0,0x1.19799812dea11p-41,0x1.19799812dea11p-40,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0000000000001p-1022},{2,2,1,2,2,0,1}},
{"points_in_poly-nan-points",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{std::numeric_limits<double>::quiet_NaN(),-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,std::numeric_limits<double>::quiet_NaN(),0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{0,0,2,3,2,3,3,3}},
{"points_in_poly-inf-points",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{std::numeric_limits<double>::infinity(),-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,-std::numeric_limits<double>::infinity(),0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{0,0,2,3,2,3,3,3}},
{"points_in_poly-nonfinite-verts",false,0,0,{0x0.0p+0,std::numeric_limits<double>::quiet_NaN(),0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,std::numeric_limits<double>::infinity(),0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{0,0,2,0,0,0,0,0}},
{"points_in_poly-dtype-bool",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0},{0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x0.0p+0},{0x1.0000000000000p+0,0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+0},{0x1.0000000000000p+0,0x1.0000000000000p+0,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x0.0p+0},{2,2,2,2,2,2,2,2}},
{"points_in_poly-dtype-int64",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"points_in_poly-dtype-half",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"points_in_poly-dtype-f32",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"points_in_poly-layout-F",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"points_in_poly-layout-reverse",false,0,0,{0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0,0x0.0p+0},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0,0x0.0p+0,-0x1.0000000000000p+0,0x1.0000000000000p+1},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+1},{3,3,3,2,3,2,0,1}},
{"points_in_poly-layout-endian",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"points_in_poly-layout-unaligned",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"points_in_poly-empty-points",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{},{},{}},
{"points_in_poly-extra-columns",false,0,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x1.0000000000000p+1,-0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+1},{0x1.0000000000000p+1,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{1,0,2,3,2,3,3,3}},
{"grid_points_in_poly-labels",true,6,7,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{},{},{2,3,3,3,2,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,2,3,3,3,2,0,0,0,0,0,0,0,0,0}},
{"grid_points_in_poly-none",true,6,7,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{},{},{2,3,3,3,2,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,2,3,3,3,2,0,0,0,0,0,0,0,0,0}},
{"grid_points_in_poly-concave",true,6,7,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+1,0x1.0000000000000p+2,0x0.0p+0},{},{},{2,3,3,3,2,0,0,3,1,1,3,0,0,0,3,1,2,0,0,0,0,3,1,1,3,0,0,0,2,3,3,3,2,0,0,0,0,0,0,0,0,0}},
{"grid_points_in_poly-closed",true,6,7,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0,0x0.0p+0},{},{},{2,3,3,3,2,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,2,3,3,3,2,0,0,0,0,0,0,0,0,0}},
{"grid_points_in_poly-clockwise",true,6,7,{0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0,0x0.0p+0},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{},{},{2,3,3,3,2,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,2,3,3,3,2,0,0,0,0,0,0,0,0,0}},
{"grid_points_in_poly-bowtie",true,6,7,{0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{},{},{2,0,0,0,2,0,0,3,3,0,3,3,0,0,3,1,0,1,3,0,0,3,3,0,3,3,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0}},
{"grid_points_in_poly-degenerate-one",true,6,7,{0x0.0p+0},{0x0.0p+0},{},{},{2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
{"grid_points_in_poly-degenerate-two",true,6,7,{0x0.0p+0,0x0.0p+0},{0x0.0p+0,0x1.0000000000000p+2},{},{},{2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
{"grid_points_in_poly-empty-rows",true,0,7,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{},{},{}},
{"grid_points_in_poly-empty-cols",true,6,0,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{},{},{}},
{"grid_points_in_poly-probe-warn",true,6,7,{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{},{},{2,3,3,3,2,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,3,1,1,1,3,0,0,2,3,3,3,2,0,0,0,0,0,0,0,0,0}},
{"grid_points_in_poly-nonfinite-verts",true,6,7,{0x0.0p+0,std::numeric_limits<double>::quiet_NaN(),0x1.0000000000000p+2,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,std::numeric_limits<double>::infinity(),0x0.0p+0},{},{},{2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0}}
};
bool same(const std::vector<double>& a,const std::vector<double>& b){
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0);
}
std::vector<std::uint8_t> point_run(const std::vector<double>& vx,const std::vector<double>& vy,
        const std::vector<double>& x,const std::vector<double>& y,int workers){
    check(vx.size()==vy.size()&&x.size()==y.size());
    auto ax=vx,ay=vy;std::vector<double> qx(x.size()+2,777.),qy(y.size()+2,-777.);
    std::copy(x.begin(),x.end(),qx.begin()+1);std::copy(y.begin(),y.end(),qy.begin()+1);
    const auto beforex=qx,beforey=qy;std::vector<std::uint8_t> out(x.size()+2,0xa5);
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const int rounding=std::fegetround();
    points_in_poly({ax.data(),ay.data(),ax.size()},qx.data()+1,qy.data()+1,x.size(),out.data()+1,workers);
    check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO&&std::fegetround()==rounding);
    check(out.front()==0xa5&&out.back()==0xa5);check(same(qx,beforex)&&same(qy,beforey)&&same(ax,vx)&&same(ay,vy));
    std::vector<std::uint8_t> result(out.begin()+1,out.end()-1);
    journal.write(reinterpret_cast<const char*>(result.data()),std::streamsize(result.size()));return result;
}
std::vector<std::uint8_t> grid_run(const std::vector<double>& vx,const std::vector<double>& vy,
        std::size_t rows,std::size_t cols,int workers){
    auto ax=vx,ay=vy;std::vector<std::uint8_t> out(rows*cols+2,0xa5);
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const int rounding=std::fegetround();
    grid_points_in_poly({ax.data(),ay.data(),ax.size()},rows,cols,out.data()+1,workers);
    check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO&&std::fegetround()==rounding);
    check(out.front()==0xa5&&out.back()==0xa5);check(same(ax,vx)&&same(ay,vy));
    std::vector<std::uint8_t> result(out.begin()+1,out.end()-1);
    journal.write(reinterpret_cast<const char*>(result.data()),std::streamsize(result.size()));return result;
}
void cohort(int workers){
    for(const auto& r:originals){
        auto result=r.grid?grid_run(r.vx,r.vy,r.rows,r.columns,workers):point_run(r.vx,r.vy,r.x,r.y,workers);
        check(result==r.expected);
    }
    const std::vector<double> vx{0,0,4,4},vy{0,4,4,0};
    const std::array<double,8> qx{2,-1,0,0,4,4,2,2},qy{2,2,0,2,4,2,4,0};
    const std::array<std::uint8_t,8> labels{1,0,2,3,2,3,3,3};
    for(std::size_t n=0;n<=33;++n){
        std::vector<double> x(n),y(n);std::vector<std::uint8_t> expected(n);
        for(std::size_t i=0;i<n;++i){x[i]=qx[i%8];y[i]=qy[i%8];expected[i]=labels[i%8];}
        check(point_run(vx,vy,x,y,workers)==expected);
    }
    // All four-lane combinations of the four independent labels.
    for(unsigned code=0;code<256;++code){
        std::vector<double> x(4),y(4);std::vector<std::uint8_t> expected(4);
        const unsigned index[4]={1,0,2,3};
        for(unsigned lane=0;lane<4;++lane){const auto label=(code>>(2*lane))&3u;expected[lane]=std::uint8_t(label);x[lane]=qx[index[label]];y[lane]=qy[index[label]];}
        check(point_run(vx,vy,x,y,workers)==expected);
    }
    // Vertex completion happens at a different loop iteration in every lane.
    check(point_run(vx,vy,{0,0,4,4},{0,4,4,0},workers)==std::vector<std::uint8_t>({2,2,2,2}));
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    check(point_run({0,0,4,4,nan},{0,4,4,0,inf},{0,0,4,4},{0,4,4,0},workers)==std::vector<std::uint8_t>({2,2,2,2}));
    const std::vector<double> special{0.,-0.,inf,-inf,nan,std::numeric_limits<double>::signaling_NaN(),1e308,-1e308};
    check(point_run({0},{0},special,std::vector<double>(special.size(),0.),workers)==std::vector<std::uint8_t>({2,2,0,0,0,0,0,0}));
    const double eps=1e-12;
    const std::vector<double> x{0,eps/2,eps,std::nextafter(eps,0.),-eps/2,-eps,std::nextafter(eps,2*eps)};
    check(point_run({0},{0},x,std::vector<double>(x.size(),0.),workers)==std::vector<std::uint8_t>({2,2,0,2,2,0,0}));
    // Simple axis-aligned polygon: independent integer geometry oracle.
    for(auto cols:{std::size_t(0),std::size_t(1),std::size_t(2),std::size_t(3),std::size_t(4),std::size_t(5),std::size_t(7),std::size_t(9)}){
        const std::size_t rows=7;std::vector<std::uint8_t> expected(rows*cols,0);
        for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c)
            if(r<=4&&c<=4)expected[r*cols+c]=(r==0||r==4)?((c==0||c==4)?2:3):((c==0||c==4)?3:1);
        check(grid_run(vx,vy,rows,cols,workers)==expected);
    }
    const std::size_t count=262151;std::vector<double> px(count),py(count);std::vector<std::uint8_t> expected(count);
    for(std::size_t i=0;i<count;++i){px[i]=qx[i%8];py[i]=qy[i%8];expected[i]=labels[i%8];}
    check(point_run(vx,vy,px,py,workers)==expected);
    const std::size_t rows=513,cols=515;expected.assign(rows*cols,0);
    for(std::size_t r=0;r<=4;++r)for(std::size_t c=0;c<=4;++c)
        expected[r*cols+c]=(r==0||r==4)?((c==0||c==4)?2:3):((c==0||c==4)?3:1);
    check(grid_run(vx,vy,rows,cols,workers)==expected);
}
template<class F>void rejected(F f){bool refused=false;try{f();}catch(const std::invalid_argument&){refused=true;}check(refused);}
void guards(){
    const double x[1]={0};std::uint8_t out=0xa5;const PolygonView good{x,x,1},empty{nullptr,nullptr,0};
    points_in_poly(empty,nullptr,nullptr,0,nullptr,1);grid_points_in_poly(empty,0,7,nullptr,1);check(out==0xa5);
    rejected([&]{points_in_poly(empty,x,x,1,&out,1);});rejected([&]{grid_points_in_poly(empty,1,1,&out,1);});
    rejected([&]{points_in_poly(good,nullptr,x,1,&out,1);});rejected([&]{grid_points_in_poly(good,1,1,nullptr,1);});
    const auto large=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    rejected([&]{grid_points_in_poly(good,large,2,&out,1);});
    rejected([&]{points_in_poly(good,x,x,large,&out,1);});
    rejected([&]{points_in_poly({x,x,large},x,x,1,&out,1);});
    for(int w:{0,1025})rejected([&]{points_in_poly(good,x,x,1,&out,w);});
    check(out==0xa5);
}
}
int main(int argc,char** argv){
    check(argc==2);journal.open(argv[1],std::ios::binary);check(bool(journal));check(skimagexx::get_num_threads()==1);
    const bool scalar=std::getenv("SKIMAGEXX_DISABLE_SIMD")!=nullptr;
    check(!scalar||!skimagexx::has_avx2());
    guards();cohort(1);
#ifdef _OPENMP
    skimagexx::set_num_threads(4);cohort(4);skimagexx::set_num_threads(1);
#else
    check(!skimagexx::has_avx2());rejected([]{skimagexx::set_num_threads(4);});
    const double x[1]={0};std::uint8_t out=0xa5;rejected([&]{points_in_poly({x,x,1},x,x,1,&out,4);});check(out==0xa5&&skimagexx::get_num_threads()==1);
#endif
    journal.close();check(bool(journal));std::printf("%zu Polygon controls PASS avx2=%d workers=%d\n",controls,int(skimagexx::has_avx2()),skimagexx::get_num_threads());
}
