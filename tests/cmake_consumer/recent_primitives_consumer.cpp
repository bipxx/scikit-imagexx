// Relocated installed-header staged APIs only: no Python or source includes.
#include <skimagexx/regular_seeds.hpp>
#include <skimagexx/unique_rows.hpp>
#include <skimagexx/label_points.hpp>
#include <skimagexx/map_array.hpp>
#include <skimagexx/level_set.hpp>
#include <skimagexx/compare_images.hpp>
#include <skimagexx/uniform_filter.hpp>
#include <skimagexx/profile_line.hpp>
#include <skimagexx/image.hpp>
#include <array>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
void check(bool value){if(!value)throw std::runtime_error("installed recent primitive mismatch");}
void stages(int workers){
    const auto global=get_num_threads(),rounding=std::fegetround(),flags=std::fetestexcept(FE_ALL_EXCEPT);
    const RegularSeedsAxis grid_axes[]={{0,2,2,3},{1,1,2,1}};
    const RegularSeedsGrid grid{grid_axes,2,4,12};std::array<std::uint32_t,12> seeds;
    seeds.fill(99);regular_seeds_assign(grid,seeds.data(),RegularSeedsType::u32,workers);
    check((seeds==std::array<std::uint32_t,12>{99,1,2,99,99,99,99,3,4,99,99,99}));
    regular_seeds_into(grid,seeds.data(),RegularSeedsType::u32,workers);
    check((seeds==std::array<std::uint32_t,12>{0,1,2,0,0,0,0,3,4,0,0,0}));

    const unsigned char rows[]={2,0,1,0,2,0,0,4};std::array<std::int64_t,4> indices{};
    check(unique_rows_indices({rows,4,2},indices.data(),workers)==3);
    check(indices[0]==3&&indices[1]==1&&indices[2]==0);

    const std::int64_t row[]={0,1,0},column[]={1,2,1};const std::int64_t* axes[]={row,column};
    const std::ptrdiff_t axis_strides[]={8,8};const std::size_t dimensions[]={2,3};
    std::array<std::uint64_t,6> labels{};
    check(label_points_into({axes,axis_strides,dimensions,2,3,6},labels.data(),workers));
    check((labels==std::array<std::uint64_t,6>{0,3,0,0,0,2}));
    labels.fill(99);
    check(label_points_assign({axes,axis_strides,dimensions,2,3,6},labels.data(),workers));
    check((labels==std::array<std::uint64_t,6>{99,3,99,99,99,2}));
    const std::int64_t invalid[]={2};const std::int64_t* bad_axes[]={invalid,column};
    std::array<std::uint64_t,6> discarded{};
    check(!label_points_into({bad_axes,axis_strides,dimensions,2,1,6},discarded.data(),workers));
    // Refused output stays private and is never consumed.

    const std::uint64_t input[]={1,2,9,1},keys[]={1,1,2};
    const std::uint16_t values[]={3,7,9};std::array<std::uint16_t,4> mapped{};
    map_array_into({input,4,keys,values,3,2,64,2},{mapped.data(),2},workers);
    check((mapped==std::array<std::uint16_t,4>{7,9,0,7}));

    const std::size_t shape[]={3,3};const LevelSetShape level{shape,2,9};
    std::array<std::int8_t,9> checker{},disk{};
    checkerboard_level_set_into(level,1,checker.data(),workers);
    check((checker==std::array<std::int8_t,9>{0,1,0,1,0,1,0,1,0}));
    const std::int64_t center[]={1,1};check(disk_level_set_geometry_admissible(level,center,1.));
    check(disk_level_set_into(level,center,1.,disk.data(),workers));
    check((disk==std::array<std::int8_t,9>{0,0,0,0,1,0,0,0,0}));

    const float first[]={1,2,3,4},second[]={4,2,1,0};std::array<float,4> comparison{};
    const CompareInput a{first,PixelType::float32},b{second,PixelType::float32};
    check(compare_images_into(a,b,comparison.data(),4,CompareMethod::diff,{},workers));
    check((comparison==std::array<float,4>{3,0,2,4}));
    check(compare_images_into(a,b,comparison.data(),4,CompareMethod::blend,{},workers));
    check((comparison==std::array<float,4>{2.5f,2,2,2}));
    check(compare_images_into(a,b,comparison.data(),4,CompareMethod::checkerboard,{2,2,1,1,2,2},workers));
    check((comparison==std::array<float,4>{1,2,1,4}));
    const float nonfinite[]={std::numeric_limits<float>::infinity()};std::array<float,1> scratch{};
    check(!compare_images_into({nonfinite,PixelType::float32},a,scratch.data(),1,CompareMethod::diff,{},workers));

    const double constant[]={7,7,7,7,7,7};std::array<double,6> filtered{};
    check(uniform_filter1d_into(constant,filtered.data(),PixelType::float64,{2,3},1,3,workers));
    check((filtered==std::array<double,6>{7,7,7,7,7,7}));

    const double line_rows[]={0,2},line_cols[]={1,3};std::array<double,12> coordinates{};
    check(profile_coordinates_into(line_rows,line_cols,2,3,1.,2.,coordinates.data(),workers));
    check((coordinates==std::array<double,12>{-1,0,1,1,2,3,-1,1,3,1,3,5}));
    const double plane[]={1,2,3,4,5,6};const double points[]={0,1,0,2};std::array<double,2> sampled{};
    check(profile_sample({plane,PixelType::float64,2,3},points,2,sampled.data(),0,GaussianBorder::nearest,0.,workers));
    check((sampled==std::array<double,2>{1,6}));
    check(get_num_threads()==global&&std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
}
}
int main(){
    check(get_num_threads()==1);stages(1);
#ifdef _OPENMP
    set_num_threads(4);stages(get_num_threads());
#endif
    set_num_threads(1);check(get_num_threads()==1);
    std::cout<<"installed recent prepared primitives PASS avx2="<<has_avx2()<<" workers=1\n";
}
