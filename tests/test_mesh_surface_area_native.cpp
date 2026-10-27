#include "mesh_surface_area.hpp"
#include "execution.hpp"
#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t controls=0;
std::ofstream journal;
void check(bool value){++controls;if(!value)std::fprintf(stderr,"mesh check %zu failed\n",controls);assert(value);}
struct Original {
    const char* id;bool single;std::size_t width;
    std::vector<double> vertices;std::vector<std::ptrdiff_t> indices;std::size_t repeats;
    std::array<double,2> expected;double allowance;
};
// BEGIN FROZEN ORIGINALS
// Saved genuine originals: journals pinned in build/native-v1/original-oracles.json.
// Layouts are normalized only for this private prepared-input ABI; no algorithm replay.
const std::vector<Original> originals={
{"right-f32",true,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.98001b1801ccap-17},
{"right-f64",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"translated",false,3, {0x1.0000000000000p+16,-0x1.0000000000000p+15,0x1.0000000000000p+14,0x1.0003000000000p+16,-0x1.0000000000000p+15,0x1.0000000000000p+14,0x1.0000000000000p+16,-0x1.fff0000000000p+14,0x1.0000000000000p+14,0x1.0001000000000p+16,-0x1.fffc000000000p+14,0x1.0008000000000p+14}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"scaled",true,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p-2,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p-1,0x0.0p+0,0x1.0000000000000p-3,0x1.0000000000000p-3,0x1.0000000000000p-2}, {0,1,2},1, {0x1.8000000000000p-4,0x1.8000000000000p-4},0x1.98001b1801ccap-23},
{"reversed-winding",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {2,1,0},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"duplicate",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},3, {0x1.2000000000000p+4,0x1.2000000000000p+4},0x1.560000000000dp-44},
{"tetrahedron",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2,0,1,3,0,2,3,1,2,3},1, {0x1.36a99b4b1f77ep+4,0x1.36a99b4b1f77ep+4},0x1.220000000000cp-43},
{"repeated-vertex",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,0,1,2,2,2},1, {0x0.0p+0,0x0.0p+0},0x0.0p+0},
{"collinear",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+1,0x1.8000000000000p+1,0x1.0000000000000p+1,0x1.0000000000000p+2,0x1.8000000000000p+2}, {0,1,2},1, {0x0.0p+0,0x0.0p+0},0x0.0p+0},
{"empty-faces",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {},1, {0x0.0p+0,0x0.0p+0},0x0.0p+0},
{"near-cancellation",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000001000p+0,0x1.fffffffffe000p-1,0x1.0000000000000p+0}, {0,1,2},1, {0x1.3988e1409212ep-40,0x1.3988e1409212ep-40},0x1.980000000000ep-47},
{"layout-F",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"layout-step",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"layout-reverse",false,3, {0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0}, {3,2,1},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"layout-endian",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"layout-unaligned",true,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.98001b1801ccap-17},
{"layout-readonly",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"layout-metadata",true,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.98001b1801ccap-17},
{"layout-faces-step",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"layout-negative-index",false,3, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {-4,-3,-2},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"intake-unused-valid",false,4, {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+1}, {0,1,2,3},1, {0x1.8000000000000p+2,0x1.8000000000000p+2},0x1.980000000000ep-46},
{"reduction-7",true,3, {0x1.0000000000000p-3,-0x1.0000000000000p-2,0x1.0000000000000p-1,0x1.6000000000000p+0,0x1.8000000000000p-1,-0x1.0000000000000p-3,-0x1.8000000000000p-2,0x1.e000000000000p+0,0x1.0000000000000p-2}, {0,1,2},7, {0x1.7c0e6c0000000p+3,0x1.7c0e6c0000000p+3},0x1.af6e26c2e57b9p-15},
{"reduction-8",true,3, {0x1.0000000000000p-3,-0x1.0000000000000p-2,0x1.0000000000000p-1,0x1.6000000000000p+0,0x1.8000000000000p-1,-0x1.0000000000000p-3,-0x1.8000000000000p-2,0x1.e000000000000p+0,0x1.0000000000000p-2}, {0,1,2},8, {0x1.b259a00000000p+3,0x1.b259a00000000p+3},0x1.0140181e0242ep-14},
{"reduction-9",true,3, {0x1.0000000000000p-3,-0x1.0000000000000p-2,0x1.0000000000000p-1,0x1.6000000000000p+0,0x1.8000000000000p-1,-0x1.0000000000000p-3,-0x1.8000000000000p-2,0x1.e000000000000p+0,0x1.0000000000000p-2}, {0,1,2},9, {0x1.e8a4d40000000p+3,0x1.e8a4d40000000p+3},0x1.2d771d70a1e00p-14},
{"reduction-128",true,3, {0x1.0000000000000p-3,-0x1.0000000000000p-2,0x1.0000000000000p-1,0x1.6000000000000p+0,0x1.8000000000000p-1,-0x1.0000000000000p-3,-0x1.8000000000000p-2,0x1.e000000000000p+0,0x1.0000000000000p-2}, {0,1,2},128, {0x1.b259a00000000p+7,0x1.b259a00000000p+7},0x1.81e0d90e7a183p-8},
{"reduction-129",true,3, {0x1.0000000000000p-3,-0x1.0000000000000p-2,0x1.0000000000000p-1,0x1.6000000000000p+0,0x1.8000000000000p-1,-0x1.0000000000000p-3,-0x1.8000000000000p-2,0x1.e000000000000p+0,0x1.0000000000000p-2}, {0,1,2},129, {0x1.b5be540000000p+7,0x1.b5be540000000p+7},0x1.8797f9cd147d3p-8},
{"reduction-8193",true,3, {0x1.0000000000000p-3,-0x1.0000000000000p-2,0x1.0000000000000p-1,0x1.6000000000000p+0,0x1.8000000000000p-1,-0x1.0000000000000p-3,-0x1.8000000000000p-2,0x1.e000000000000p+0,0x1.0000000000000p-2}, {0,1,2},8193, {0x1.b266ae0000000p+13,0x1.b267340000000p+13},0x1.57ec0a09ade07p+4},
};
// END FROZEN ORIGINALS
template<class T>constexpr ColocType type(){return sizeof(T)==4?ColocType::float32:ColocType::float64;}
template<class T>T value(const ColocScalar& result){T out;std::memcpy(&out,&result.bits,sizeof(T));return out;}
template<class T>ColocScalar run(const std::vector<T>& input,std::size_t width,ColocReductionPlan plan,int workers){
    std::vector<T> guarded(input.size()+2,T(31.25));std::copy(input.begin(),input.end(),guarded.begin()+1);
    const auto before=guarded;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    const auto result=mesh_surface_area({guarded.data()+1,type<T>(),input.size()/(width*3),width},plan,workers);
    check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    check(std::memcmp(guarded.data(),before.data(),guarded.size()*sizeof(T))==0);
    check(result.type==type<T>()&&result.fp_status==0);
    journal.write(reinterpret_cast<const char*>(&result.bits),sizeof(result.bits));return result;
}
template<class T>void original(const Original& row,int workers){
    std::vector<T> gathered;
    for(std::size_t r=0;r<row.repeats;++r)for(auto index:row.indices){
        if(index<0)index+=std::ptrdiff_t(row.vertices.size()/3);
        for(std::size_t j=0;j<3;++j)gathered.push_back(T(row.vertices[std::size_t(index)*3+j]));
    }
    std::size_t profile=0;
    for(auto mode:{ColocProfile::numpy_1244,ColocProfile::numpy_254}){
        const auto result=run(gathered,row.width,{mode,std::string(row.id)=="reduction-8193"?16u:8192u},workers);
        // Source-derived scalar /2 only; the saved final original is the oracle.
        const double area=double(value<T>(result))/2.;
        check(std::abs(area-row.expected[profile++])<=row.allowance);
        if(row.allowance==0)check(result.bits==0);
    }
}
template<class T>std::vector<T> triangles(std::size_t count,std::size_t width){
    std::vector<T> input(count*width*3,T(0));
    for(std::size_t i=0;i<count;++i){
        input[i*width*3+3]=3;input[i*width*3+7]=4;
        if(width>3)input[i*width*3+9]=std::numeric_limits<T>::quiet_NaN(); // Gathered but unused.
    }
    return input;
}
template<class T>void tails(int workers){
    for(auto n:{std::size_t(0),std::size_t(1),std::size_t(3),std::size_t(4),std::size_t(5),std::size_t(7),std::size_t(8),std::size_t(9),
                std::size_t(127),std::size_t(128),std::size_t(129),std::size_t(8191),std::size_t(8192),std::size_t(8193)})
    for(std::size_t width:{3u,4u}){
        const auto input=triangles<T>(n,width);
        for(auto mode:{ColocProfile::numpy_1244,ColocProfile::numpy_254})for(std::size_t buffer:{16u,8192u})
            check(value<T>(run(input,width,{mode,buffer},workers))==T(n*12));
    }
    check(value<T>(run(triangles<T>(262151,3),3,{},workers))==T(262151*12));
    // Known exact face norms exercise both existing profile-specific sum plans.
    const std::size_t n=8193;auto input=triangles<T>(n,3);std::vector<T> norms(n,T(1));
    for(std::size_t i=0;i<n;++i){input[i*9+3]=1;input[i*9+7]=1;}
    norms[0]=std::ldexp(T(1),std::numeric_limits<T>::digits);input[7]=norms[0];
    const std::ptrdiff_t stride=sizeof(T);
    ColocView view{norms.data(),norms.data(),norms.size()*sizeof(T),{&n,1},{&stride,1},type<T>(),false};
    for(auto mode:{ColocProfile::numpy_1244,ColocProfile::numpy_254})for(std::size_t buffer:{16u,8192u}){
        const auto expected=coloc_sum(view,{mode,buffer});
        check(run(input,3,{mode,buffer},workers).bits==expected.bits);
    }
}
template<class T>void refused(const std::vector<T>& input,int workers){
    const auto before=input;std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    bool caught=false;
    try{mesh_surface_area({input.data(),type<T>(),input.size()/9,3},{},workers);}
    catch(const ColocProviderRequired&){caught=true;}
    check(caught);check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    check(std::memcmp(input.data(),before.data(),input.size()*sizeof(T))==0);
    journal.put(char(caught));
}
template<class T>void exceptions(int workers){
    for(auto n:{std::size_t(1),std::size_t(9)}){
        for(T bad:{std::numeric_limits<T>::quiet_NaN(),std::numeric_limits<T>::infinity(),-std::numeric_limits<T>::infinity()}){
            auto input=triangles<T>(n,3);input[(n-1)*9]=bad;refused(input,workers);
        }
        auto input=triangles<T>(n,3);input[0]=std::numeric_limits<T>::max();input[3]=-std::numeric_limits<T>::max();refused(input,workers);
        // Normal operands can subtract to an EXACT subnormal without an FP flag.
        input=triangles<T>(n,3);input[0]=std::numeric_limits<T>::min();input[3]=std::nextafter(input[0],T(1));refused(input,workers);
        // Exact subnormal cross product; then exact subnormal square, separately.
        for(int exponent:{sizeof(T)==4?-70:-530,sizeof(T)==4?-35:-265}){
            input=triangles<T>(n,3);input[3]=input[7]=std::ldexp(T(1),exponent);refused(input,workers);
        }
        input=triangles<T>(n,3);input[3]=input[7]=std::ldexp(T(1),sizeof(T)==4?-100:-600);refused(input,workers);
        input=triangles<T>(n,3);input[3]=input[7]=std::ldexp(T(1),sizeof(T)==4?40:300);refused(input,workers);
    }
    auto large=triangles<T>(262151,3);large[large.size()-2]=std::numeric_limits<T>::infinity();refused(large,workers);
    auto input=triangles<T>(9,3);
    const auto rounding=std::fegetround();check(std::fesetround(FE_DOWNWARD)==0);refused(input,workers);
    check(std::fegetround()==FE_DOWNWARD);check(std::fesetround(rounding)==0);
}
void cohort(int workers){
    for(const auto& row:originals)if(row.single)original<float>(row,workers);else original<double>(row,workers);
    tails<float>(workers);tails<double>(workers);exceptions<float>(workers);exceptions<double>(workers);
}
void structural(){
    const auto input=triangles<float>(1,3);
    for(int workers:{0,1025}){bool caught=false;try{mesh_surface_area({input.data(),ColocType::float32,1,3},{},workers);}catch(const std::invalid_argument&){caught=true;}check(caught);}
    for(const auto bad:{MeshAreaInput{nullptr,ColocType::float32,1,3},MeshAreaInput{input.data(),ColocType::float32,1,2},MeshAreaInput{input.data(),ColocType::int32,1,3}}){
        bool caught=false;try{mesh_surface_area(bad,{},1);}catch(const std::invalid_argument&){caught=true;}check(caught);
    }
    bool caught=false;try{mesh_surface_area({input.data(),ColocType::float32,1,std::size_t(std::numeric_limits<std::ptrdiff_t>::max())},{},1);}catch(const std::overflow_error&){caught=true;}check(caught);
    caught=false;try{mesh_surface_area({input.data(),ColocType::float32,(1u<<22)-16,3},{},1);}catch(const ColocProviderRequired&){caught=true;}check(caught);
}
}
int main(int argc,char** argv){
    check(argc==2);journal.open(argv[1],std::ios::binary);check(bool(journal));
    check(get_num_threads()==1);structural();cohort(1);
#ifdef _OPENMP
    set_num_threads(4);cohort(4);set_num_threads(1);
#else
    bool caught=false;try{set_num_threads(4);}catch(const std::invalid_argument& error){caught=std::string(error.what())=="this build has no OpenMP support";}
    check(caught&&get_num_threads()==1);
    caught=false;try{mesh_surface_area({nullptr,ColocType::float32,0,3},{},4);}catch(const std::invalid_argument&){caught=true;}check(caught);
#endif
    journal.close();check(bool(journal));
    std::printf("%zu Mesh controls PASS avx2=%d workers=%d\n",controls,int(has_avx2()),get_num_threads());
}
