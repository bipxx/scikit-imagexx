#include "skimagexx/component_consumers.hpp"
#include "cmake_consumer/find_boundaries_consumer.hpp"
#include "cmake_consumer/join_segmentations_consumer.hpp"
#include "cmake_consumer/regular_grid_consumer.hpp"
#include "cmake_consumer/crop_component_consumer.hpp"
#include "skimagexx/image.hpp"
#include <skimagexx/regionprops_state.hpp>
#include <string>
#include <algorithm>
#include <array>
#include <atomic>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
using namespace skimagexx;
namespace {
void require(bool value){if(!value)throw std::runtime_error("component consumer native assertion failed");}
std::size_t region_state_checks=0;
// Installed SDK composition: visible population, separately visible box size,
// and a cached area value are deliberately different state inputs.

// Existing component-consumer family, also reached by its no-argument CTest.
void regionprops_coords_scaled(int workers) {
    using namespace skimagexx::detail;
    auto check=[](bool v){++region_state_checks;require(v);};
    const int previous=get_num_threads();set_num_threads(workers);
    std::fenv_t environment;check(std::fegetenv(&environment)==0);
    check(std::fesetround(FE_TONEAREST)==0);
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    const std::array<std::size_t,3> shape{2,2,1};
    const std::array<std::intptr_t,3> origin{1,1,0};
    const std::array<double,3> spacing{-.5,2.,0.},offset{-3.,4.,0.};
    // Actual frozen negative-anisotropic2d words, F permutation only.
    const std::array<double,6> expected{-3.5,-3.5,-4.,6.,8.,6.};
    for(int layout=0;layout<3;++layout) {
        std::uint8_t storage[6]={0x91,1,1,1,0,0x91};auto* mask=storage+1;
        std::array<std::ptrdiff_t,3> strides{2,1,0};
        if(layout==1){strides={1,2,0};}
        if(layout==2){mask=storage+4;strides={-2,-1,0};storage[1]=0;storage[4]=1;}
        const std::vector<std::uint8_t> input(storage,storage+6);
        std::intptr_t indices[8];std::fill(indices,indices+8,-91);double output[8];std::fill(output,output+8,-97.);
        check(regionprops_coords_scaled_local_into(mask,2,shape,strides,3,indices+1,workers));
        const std::intptr_t local[]{0,0,1,0,1,0};for(std::size_t i=0;i<6;++i)check(indices[i+1]==local[i]);
        check(regionprops_coords_scaled_origin_into(indices+1,3,2,shape,origin,workers));
        check(regionprops_coords_scaled_multiply_f64_into(indices+1,3,2,spacing,output+1,workers));
        check(regionprops_coords_scaled_add_f64_into(output+1,3,2,offset,workers));
        for(std::size_t i=0;i<6;++i)check(std::memcmp(output+1+i,&expected[i],sizeof(double))==0);
        check(output[0]==-97.&&output[7]==-97.&&indices[0]==-91&&indices[7]==-91);
        check(std::equal(input.begin(),input.end(),storage));check(std::fetestexcept(FE_ALL_EXCEPT)==flags);
    }
    // CURRENT edited mask count, independent of constructor/cached population.
    const std::uint8_t edited[]{0,1,1,0};std::intptr_t indices[4];double output[4];
    check(regionprops_coords_scaled_local_into(edited,2,shape,{2,1,0},2,indices,workers));
    check(regionprops_coords_scaled_origin_into(indices,2,2,shape,origin,workers));
    check(regionprops_coords_scaled_multiply_f64_into(indices,2,2,spacing,output,workers));
    check(regionprops_coords_scaled_add_f64_into(output,2,2,offset,workers));
    const double edited_expected[]{-3.5,-4.,8.,6.};check(std::memcmp(output,edited_expected,sizeof(output))==0);
    const std::array<std::size_t,3> empty_shape{0,2,1};
    check(regionprops_coords_scaled_local_into(nullptr,2,empty_shape,{2,1,0},0,nullptr,workers));
    check(regionprops_coords_scaled_origin_into(nullptr,0,2,empty_shape,origin,workers));
    check(regionprops_coords_scaled_multiply_f64_into(nullptr,0,2,spacing,nullptr,workers));
    check(regionprops_coords_scaled_add_f64_into(nullptr,0,2,offset,workers));
    std::intptr_t invalid[]{0,1,std::numeric_limits<std::intptr_t>::max(),0};
    const std::vector<std::intptr_t> before(invalid,invalid+4);
    check(!regionprops_coords_scaled_origin_into(invalid,2,2,shape,origin,workers));check(std::equal(before.begin(),before.end(),invalid));
    const double untouched[4]={1.,2.,3.,4.};std::copy(untouched,untouched+4,output);
    auto invalid_spacing=spacing;const std::uint64_t snan=UINT64_C(0x7ff0000000000001);std::memcpy(&invalid_spacing[1],&snan,8);
    check(!regionprops_coords_scaled_multiply_f64_into(indices,2,2,invalid_spacing,output,workers));check(std::memcmp(output,untouched,sizeof(output))==0);check(std::fetestexcept(FE_ALL_EXCEPT)==flags);
    check(std::fesetround(FE_DOWNWARD)==0);
    check(!regionprops_coords_scaled_multiply_f64_into(indices,2,2,spacing,output,workers));check(std::memcmp(output,untouched,sizeof(output))==0);check(std::fegetround()==FE_DOWNWARD);
    check(!regionprops_coords_scaled_add_f64_into(output,2,2,offset,workers));check(std::memcmp(output,untouched,sizeof(output))==0);
    check(std::fesetenv(&environment)==0);set_num_threads(previous);check(get_num_threads()==previous);
}

void regionprops_scalars(int workers) {
    using namespace skimagexx::detail;
    auto check=[](bool value){++region_state_checks;require(value);};
    auto number=[](RegionPropsScalar64 value){double out;std::memcpy(&out,&value.bits,8);return out;};
    std::fenv_t original;check(std::fegetenv(&original)==0);
    check(std::fesetround(FE_TONEAREST)==0);
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INEXACT);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    set_num_threads(workers);
    std::array<std::uint8_t,6> visible{1,0,1,1,0,1};
    const auto area=regionprops_scaled_count_f64(regionprops_mask_population(visible.data(),visible.size()),.75);
    const auto box=regionprops_scaled_count_f64(visible.size(),.75);
    check(area.bits==0x4008000000000000ULL&&area.fp_status==0);
    check(box.bits==0x4012000000000000ULL&&box.fp_status==0);
    const auto extent=regionprops_extent_f64(number(area),number(box));
    check(extent.bits==0x3fe5555555555555ULL&&extent.fp_status==0);
    visible={1,0,0,0,0,0};
    const auto fresh=regionprops_scaled_count_f64(regionprops_mask_population(visible.data(),visible.size()),.75);
    check(fresh.bits==0x3fe8000000000000ULL&&fresh.fp_status==0);
    check(regionprops_extent_f64(number(fresh),number(box)).bits==0x3fc5555555555555ULL);
    // A retained cached area is not silently recomputed from the edited mask.
    check(regionprops_extent_f64(number(area),number(box)).bits==extent.bits);
    const auto resized_box=regionprops_scaled_count_f64(2,.75);
    check(regionprops_extent_f64(number(area),number(resized_box)).bits==0x4000000000000000ULL);
    const auto empty=regionprops_scaled_count_f64(0,.75);
    check(empty.bits==0&&empty.fp_status==0);
    const auto invalid=regionprops_extent_f64(number(empty),number(empty));
    check(invalid.bits==0xfff8000000000000ULL&&invalid.fp_status==regionprops_fp_invalid);
    const auto divide=regionprops_extent_f64(number(area),0.);
    check(divide.bits==0x7ff0000000000000ULL&&divide.fp_status==regionprops_fp_divide);
    check(std::fegetround()==FE_TONEAREST&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
    check(get_num_threads()==workers);
    check(std::fesetenv(&original)==0);
}
void regionprops_state(int workers) {
    using namespace skimagexx::detail;
    auto check=[](bool value){++region_state_checks;require(value);};
    set_num_threads(workers);
    std::array<std::int16_t,12> labels{-1,5,0,2,5,0,2,0,0,5,0,2};
    ComponentsView view{labels.data(),PixelType::int16,false,false,{3,4},{8,2}};
    RegionPropsState state;check(prepare_regionprops_state(view,state));
    check(state.workers==workers&&state.labels.data==labels.data()&&state.regions.size()==2);
    const auto bounds=state.regions[0];
    check(bounds.label==2&&bounds.initial_population==3&&bounds.start==std::array<std::size_t,3>{0,2,0}&&bounds.stop==std::array<std::size_t,3>{3,4,0});
    check(state.regions[1].label==5&&state.regions[1].initial_population==3);
    std::array<std::uint8_t,8> mask;mask.fill(91);
    check(regionprops_mask_into(state,0,mask.data()+1)==3);
    check((mask==std::array<std::uint8_t,8>{91,0,1,1,0,0,1,91}));
    std::array<std::intptr_t,8> coords;coords.fill(-999);
    regionprops_coords_into(state,0,mask.data()+1,coords.data()+1);
    check((coords==std::array<std::intptr_t,8>{-999,0,3,1,2,2,3,-999}));
    labels[3]=0;labels[1]=2; // New hit outside frozen bbox is deliberately excluded.
    check(regionprops_mask_into(state,0,mask.data()+1)==2);
    check(state.regions[0].start==bounds.start&&state.regions[0].stop==bounds.stop&&state.regions[0].initial_population==3);
    check(regionprops_mask_population(mask.data()+1,6)==2&&mask.front()==91&&mask.back()==91);
    coords.fill(-999);regionprops_coords_into(state,0,mask.data()+1,coords.data()+1);
    check((coords==std::array<std::intptr_t,8>{-999,1,2,2,3,-999,-999,-999}));
    mask[1]=7; // Cached-mask edits use nonzero truth, not initial label population.
    check(regionprops_mask_population(mask.data()+1,6)==3);
    coords.fill(-999);regionprops_coords_into(state,0,mask.data()+1,coords.data()+1);
    check((coords==std::array<std::intptr_t,8>{-999,0,2,1,2,2,3,-999}));
    std::array<std::int16_t,6> reversed{99,1,0,3,1,99};const auto before=reversed;
    ComponentsView negative{reversed.data()+4,PixelType::int16,false,false,{2,2},{-4,-2}};
    RegionPropsState signed_state;check(prepare_regionprops_state(negative,signed_state));
    check(signed_state.regions.size()==2&&signed_state.regions[0].label==1&&signed_state.regions[1].label==3);
    std::array<std::uint8_t,6> signed_mask;signed_mask.fill(91);
    check(regionprops_mask_into(signed_state,0,signed_mask.data()+1)==2);
    check((signed_mask==std::array<std::uint8_t,6>{91,1,0,0,1,91}));
    coords.fill(-999);regionprops_coords_into(signed_state,0,signed_mask.data()+1,coords.data()+1);
    check((coords==std::array<std::intptr_t,8>{-999,0,0,1,1,-999,-999,-999}));
    check(reversed==before);
    for(int fault=0;fault<6;++fault){
        auto bad=view;
        if(fault==0)bad.type=PixelType::boolean;
        if(fault==1)bad.swap=true;
        if(fault==2)bad.half=true;
        if(fault==3)bad.data=nullptr;
        if(fault==4)bad.shape={0,4};
        if(fault==5)bad.strides={std::numeric_limits<std::ptrdiff_t>::min(),2};
        check(!prepare_regionprops_state(bad,state));
        check(state.labels.data==labels.data()&&state.regions[0].start==bounds.start&&state.regions[0].stop==bounds.stop);
    }
    // execution.hpp parallel_min is256*1024;4*65537 exceeds it by4.
    constexpr std::size_t width=65537,n=4*width;
    std::vector<std::int16_t> large(n,7);
    ComponentsView large_view{large.data(),PixelType::int16,false,false,{4,width},{std::ptrdiff_t(width*2),2}};
    RegionPropsState large_state;check(prepare_regionprops_state(large_view,large_state));
    check(large_state.workers==workers&&large_state.regions.size()==1&&large_state.regions[0].initial_population==n);
    // Existing states must honor a later per-operation global-budget change.
#ifdef _OPENMP
    const int mask_workers=workers==1?4:1;
#else
    const int mask_workers=1;
#endif
    set_num_threads(mask_workers);
    check(large_state.workers==workers);
    large[0]=0;
    std::vector<std::uint8_t> large_mask(n+2,91);
    check(regionprops_mask_into(large_state,0,large_mask.data()+1)==n-1);
    check(large_mask.front()==91&&large_mask.back()==91&&large_mask[1]==0);
    check(regionprops_mask_population(large_mask.data()+1,n)==n-1);
    std::vector<std::intptr_t> large_coords(2*(n-1)+2,-999);
    regionprops_coords_into(large_state,0,large_mask.data()+1,large_coords.data()+1);
    check(large_coords.front()==-999&&large_coords.back()==-999);
    for(std::size_t i=1;i<n;++i){check(large_mask[i+1]==1);check(large_coords[2*(i-1)+1]==std::intptr_t(i/width)&&large_coords[2*(i-1)+2]==std::intptr_t(i%width));}
    check(get_num_threads()==mask_workers);
    set_num_threads(workers);
    check(get_num_threads()==workers);
}
void regionprops_strided(int workers) {
    using namespace skimagexx::detail;
    auto check=[](bool value){++region_state_checks;require(value);};
    set_num_threads(workers);
    std::array<std::int16_t,12> labels{-1,5,0,2,5,0,2,0,0,5,0,2};
    ComponentsView view{labels.data(),PixelType::int16,false,false,{3,4},{8,2}};
    RegionPropsState state;check(prepare_regionprops_state(view,state));
    std::array<std::uint8_t,8> mask;mask.fill(91);
    check(regionprops_mask_strided_into(state,0,mask.data()+1,{1,3,0})==3);
    check((mask==std::array<std::uint8_t,8>{91,0,1,0,1,0,1,91}));
    std::array<std::intptr_t,8> output;output.fill(-999);
    regionprops_coords_strided_into(state,0,mask.data()+1,{1,3,0},output.data()+1,sizeof(std::intptr_t),3*sizeof(std::intptr_t));
    check((output==std::array<std::intptr_t,8>{-999,0,1,2,3,2,3,-999}));
    labels[3]=0; // Current pixels change, frozen constructor box does not.
    check(regionprops_mask_strided_into(state,0,mask.data()+6,{-1,-3,0})==2);
    check(mask.front()==91&&mask.back()==91);
    mask[6]=7; // Current supplied mask, including noncanonical nonzero truth.
    output.fill(-999);
    regionprops_coords_strided_into(state,0,mask.data()+6,{-1,-3,0},output.data()+6,-2*std::ptrdiff_t(sizeof(std::intptr_t)),sizeof(std::intptr_t));
    check((output==std::array<std::intptr_t,8>{-999,-999,2,3,1,2,0,2}));
    check(state.regions[0].initial_population==3&&state.regions[0].start==std::array<std::size_t,3>{0,2,0});
    // Three-dimensional Fortran storage with exact canonical coordinate order.
    std::vector<std::uint16_t> volume(2*3*4,9);
    ComponentsView cube{volume.data(),PixelType::uint16,false,false,{2,3,4},{2,4,12}};
    RegionPropsState cube_state;check(prepare_regionprops_state(cube,cube_state));
    std::vector<std::uint8_t> cube_mask(volume.size()+2,91);volume[0]=0;
    check(regionprops_mask_strided_into(cube_state,0,cube_mask.data()+1,{1,2,6})==23);
    std::vector<std::intptr_t> cube_coords(23*3+2,-999);
    regionprops_coords_strided_into(cube_state,0,cube_mask.data()+1,{1,2,6},cube_coords.data()+1,sizeof(std::intptr_t),23*sizeof(std::intptr_t));
    for(std::size_t i=1;i<volume.size();++i){
        check(cube_coords[i]==std::intptr_t(i/12));
        check(cube_coords[23+i]==std::intptr_t((i/4)%3));
        check(cube_coords[46+i]==std::intptr_t(i%4));
    }
    check(cube_mask.front()==91&&cube_mask.back()==91&&cube_coords.front()==-999&&cube_coords.back()==-999);
    // Source-safe analytic span just above existing parallel_min, F input/output.
    constexpr std::size_t width=65537,n=4*width;
    std::vector<std::int16_t> large(n,7);
    ComponentsView large_view{large.data(),PixelType::int16,false,false,{4,width},{2,8}};
    RegionPropsState large_state;check(prepare_regionprops_state(large_view,large_state));
#ifdef _OPENMP
    const int current=workers==1?4:1;
#else
    const int current=1;
#endif
    set_num_threads(current);large[0]=0;
    std::vector<std::uint8_t> large_mask(n+2,91);
    check(regionprops_mask_strided_into(large_state,0,large_mask.data()+1,{1,4,0})==n-1);
    std::vector<std::intptr_t> large_coords(2*(n-1)+2,-999);
    regionprops_coords_strided_into(large_state,0,large_mask.data()+1,{1,4,0},large_coords.data()+1,sizeof(std::intptr_t),(n-1)*sizeof(std::intptr_t));
    for(std::size_t i=1;i<n;++i){
        check(large_mask[1+(i/width)+4*(i%width)]==1);
        check(large_coords[i]==std::intptr_t(i/width));
        check(large_coords[n-1+i]==std::intptr_t(i%width));
    }
    check(large_mask.front()==91&&large_mask.back()==91&&large_mask[1]==0);
    check(large_coords.front()==-999&&large_coords.back()==-999);
    check(large_state.workers==workers&&get_num_threads()==current);
    set_num_threads(workers);check(get_num_threads()==workers);
}
void count_filter(int workers) {
    set_num_threads(workers);constexpr std::size_t n=513*513;
    std::vector<std::intptr_t> labels(n);std::array<std::intptr_t,8> expected{};
    for(std::size_t i=0;i<n;++i){labels[i]=std::intptr_t(i%8);++expected[i%8];}
    auto counts=consumer_count(labels.data(),n);require(!counts.empty && counts.maximum==7);
    require(std::equal(expected.begin(),expected.end(),counts.sizes.begin()));
    std::vector<std::int16_t> pixels(n+2,-99);for(std::size_t i=0;i<n;++i)pixels[n-i]=std::int16_t(labels[i]);
    ConsumerRaster output{pixels.data()+n,{n},{-2},2};
    const double threshold=double(n/8);consumer_filter_into(labels.data(),counts,threshold,output);
    for(std::size_t i=0;i<n;++i)require(pixels[n-i]==(double(expected[i%8])<=threshold ? 0:std::int16_t(labels[i])));
    require(pixels.front()==-99 && pixels.back()==-99);
    std::vector<std::uint8_t> border(n);consumer_border_into(border.data(),{513,513},{1,1},{512,512});
    for(std::size_t y=0;y<513;++y)for(std::size_t x=0;x<513;++x)require(bool(border[y*513+x])==(x==0 || y==0 || x==512 || y==512));
    auto remove=consumer_border_ids(labels.data(),border.data(),n,7);require(std::all_of(remove.begin(),remove.end(),[](auto x){return x==1;}));
    const std::int16_t replacement=11;consumer_clear_into(labels.data(),remove.data(),7,&replacement,output);
    for(std::size_t i=1;i<=n;++i)require(pixels[i]==11);
}
#ifdef _OPENMP
void concurrent_configuration() {
    constexpr std::size_t n=513*513;
    std::vector<std::intptr_t> labels(n,1),pixels(n);
    ComponentsView input;input.data=labels.data();input.type=PixelType::int64;input.shape={n};input.strides={8};
    ConsumerRaster output{pixels.data(),{n},{8},8};const std::uint8_t remove[]{0,1};const std::intptr_t fill=7;
    std::atomic<bool> stop{false};std::thread setter([&]{while(!stop.load(std::memory_order_relaxed)){set_num_threads(1);set_num_threads(4);}});
    try {
        for(int repeat=0;repeat<20;++repeat) {
            require(!consumer_min_negative(input));
            consumer_clear_into(labels.data(),remove,1,&fill,output);
            require(std::all_of(pixels.begin(),pixels.end(),[](auto value){return value==7;}));
        }
    } catch(...) {stop.store(true);setter.join();throw;}
    stop.store(true);setter.join();set_num_threads(1);
}
#endif
}
int main(int argc,char** argv) {
    if(argc==1)for(int workers:{1,4}){
        if(workers==4){try{set_num_threads(4);}catch(const std::invalid_argument&){require(get_num_threads()==1);continue;}set_num_threads(1);}
        require(skimagexx_test::find_boundaries_consumer(workers)>0);
        require(skimagexx_test::join_segmentations_consumer(workers)>0);
        require(skimagexx_test::regular_grid_consumer(workers)>0);
        require(skimagexx_test::crop_component_consumer(workers)>0);
    }
    require(get_num_threads()==1);
    require(argc==1||(argc==2&&(std::string(argv[1])=="--regionprops-state-only"||std::string(argv[1])=="--regionprops-scalar-only"||std::string(argv[1])=="--regionprops-strided-only"||std::string(argv[1])=="--regionprops-coords-scaled-only")));
    if(argc==1||(argc==2&&std::string(argv[1])=="--regionprops-coords-scaled-only")){
        for(int workers:{1,4}) {
            if(workers==4){try{set_num_threads(4);}catch(const std::invalid_argument&){require(get_num_threads()==1);continue;}set_num_threads(1);}
            regionprops_coords_scaled(workers);
        }
        set_num_threads(1);require(get_num_threads()==1);
        if(argc==2){std::cout<<"installed coords_scaled controls="<<region_state_checks<<"; default="<<get_num_threads()<<"; AVX2="<<has_avx2()<<"\n";return 0;}
    }
    if(argc==2&&std::string(argv[1])=="--regionprops-strided-only"){
        const bool available=has_avx2();
        for(int workers:{1
#ifdef _OPENMP
            ,4
#endif
        })regionprops_strided(workers);
        set_num_threads(1);require(get_num_threads()==1);
        std::cout<<"installed regionprops strided controls="<<region_state_checks<<"; default="<<get_num_threads()<<"; AVX2="<<available<<"\n";
        return 0;
    }
    if(argc==1||std::string(argv[1])=="--regionprops-state-only")
        for(int workers:{1
#ifdef _OPENMP
            ,4
#endif
        })regionprops_strided(workers);
    for(int workers:{1
#ifdef _OPENMP
        ,4
#endif
    })regionprops_scalars(workers);
    if(argc==2&&std::string(argv[1])=="--regionprops-scalar-only"){
        set_num_threads(1);
        std::cout<<"installed regionprops scalar controls="<<region_state_checks<<"; default="<<get_num_threads()<<"; AVX2="<<has_avx2()<<"\n";
        return 0;
    }
    for(int workers:{1
#ifdef _OPENMP
        ,4
#endif
    })regionprops_state(workers);
#ifndef _OPENMP
    bool refused=false;try{set_num_threads(4);}catch(const std::invalid_argument&){refused=true;}
    require(refused&&get_num_threads()==1);
#endif
    set_num_threads(1);require(get_num_threads()==1);
    if(argc==2){std::cout<<"installed regionprops state controls="<<region_state_checks<<"; available worker controls/default1; AVX2="<<has_avx2()<<"\n";return 0;}
    require(consumer_count(nullptr,0).empty);
    const std::intptr_t sparse[]{0,1000000000000LL,3,1000000000000LL};
    auto counts=consumer_count(sparse,4);require(counts.keys==std::vector<std::uint64_t>({0,3,1000000000000ULL}));
    require(counts.sizes==std::vector<std::intptr_t>({1,1,2}));
    const std::intptr_t negative[]{-1,0};bool rejected=false;try{consumer_count(negative,2);}catch(const std::invalid_argument&){rejected=true;}require(rejected);
    std::array<std::intptr_t,33> truth{};std::array<std::uint8_t,33> bytes{},inverted{};
    for(std::size_t i=0;i<truth.size();++i)truth[i]=std::intptr_t(i%3)-1;
    consumer_truth_into(truth.data(),bytes.data(),truth.size(),false);
    ConsumerRaster boolean{inverted.data(),{33},{1},1};consumer_bool_write(bytes.data(),boolean,true);
    for(std::size_t i=0;i<truth.size();++i){require(bytes[i]==(truth[i]!=0));require(inverted[i]==(truth[i]==0));}
    std::vector<std::uint16_t> half(65536);std::vector<std::uint8_t> nonzero(65536);
    for(std::size_t i=0;i<half.size();++i)half[i]=std::uint16_t(i);
    ComponentsView view;view.data=half.data();view.type=PixelType::uint16;view.half=true;view.shape={half.size()};view.strides={2};
    consumer_nonzero_into(view,nonzero.data());for(std::size_t i=0;i<half.size();++i)require(bool(nonzero[i])==bool(i&0x7fff));
    require(!consumer_min_negative(view));
    const std::uint64_t raw[]{std::numeric_limits<std::uint64_t>::max(),3,0,1000000000000ULL,3};
    auto unique=consumer_sorted_unique(raw,5);require(unique==std::vector<std::uint64_t>({0,3,1000000000000ULL,std::numeric_limits<std::uint64_t>::max()}));
    const std::intptr_t invalid_labels[]{-1,0};const std::uint8_t invalid_border[]{1,0};rejected=false;try{consumer_border_ids(invalid_labels,invalid_border,2,1);}catch(const std::invalid_argument&){rejected=true;}require(rejected);
    const std::uint64_t keys[]{0,3,1000000000000ULL},index[]{3,1000000000000ULL,99,0};const std::int16_t values[]{0,7,9};std::array<std::int16_t,4> mapped{};
    consumer_map_into(index,4,keys,values,3,2,64,mapped.data());require(mapped==std::array<std::int16_t,4>{7,9,0,0});
    std::array<std::int16_t,4> dense{};const std::uint64_t dense_keys[]{1,3};const std::int16_t dense_values[]{5,8};
    consumer_dense_into(dense_keys,dense_values,2,2,dense.data(),4);require(dense==std::array<std::int16_t,4>{0,5,0,8});
    const std::uint64_t scatter[]{0,2,2};const std::int16_t updates[]{1,2,3};consumer_scatter_into(scatter,updates,3,2,dense.data(),4);require(dense==std::array<std::int16_t,4>{1,5,3,8});
    for(int workers:{1,
#ifdef _OPENMP
        4
#else
        1
#endif
    }) {
        count_filter(workers);
        std::vector<std::uint64_t> large(300001);for(std::size_t i=0;i<large.size();++i)large[i]=1000000000000ULL+(i*17)%257;
        const auto result=consumer_sorted_unique(large.data(),large.size());require(result.size()==257);
        for(std::size_t i=0;i<result.size();++i)require(result[i]==1000000000000ULL+i);
    }
#ifdef _OPENMP
    concurrent_configuration();
#endif
    set_num_threads(1);std::cout<<"component consumer native contracts passed\n";
}
