// Fixed, versioned general finite cohorts. The independent NumPy/original
// oracle reads this JSON verbatim; these are named data, not random fuzz.
// MATRIX_COHORTS_BEGIN
/*
{"schema":"matrix-general-finite-cohorts-v2","include_literal_base":true,"rows":[1,2,3,4,7,8,9,15,16,17,31,32,33,63,64],"layouts":["C","F","reverse","step","unaligned"],"row_extension":0.03125,"cases":[
{"id":"large-image-coordinates","coords":[[1024.25,512.5],[4095.75,2047.25]],"matrix":[[1,0,320],[0,1,-240],[0,0,1]]},
{"id":"ordinary-affine-coefficients","coords":[[0.123456789,-0.34567891],[1234.567,2345.678]],"matrix":[[0.9987623,-0.03456789,12.3456],[0.04567891,1.0012345,-7.6543],[0,0,1]]},
{"id":"ordinary-projective-coefficients","coords":[[31.125,15.875],[1023.75,511.25]],"matrix":[[0.9937,0.0123,15.67],[-0.0234,1.0045,7.89],[0.0001234,-0.0002345,1]]},
{"id":"ordinary-3D-transform","coords":[[16.125,32.25,64.5],[-128.125,256.25,512.5]],"matrix":[[1.000123,0,0,1.23],[0,0.999876,0,-2.34],[0,0,1.000345,3.45],[0.0000123,-0.0000234,0.0000345,1]]},
{"id":"ordinary-4D-transform","coords":[[0.1234567,-31.23456,511.34567,2047.4567],[-17.56789,127.67891,-255.78912,4095.89123]],"matrix":[[1.001234,0.002345,-0.003456,0.004567,12.3456],[-0.005678,0.998765,0.006789,-0.007891,-23.4567],[0.008912,-0.009123,1.002345,0.001234,34.5678],[-0.002345,0.003456,-0.004567,0.997654,-45.6789],[0.0000123,-0.0000234,0.0000345,-0.0000456,1]]},
{"id":"fma-cancellation-2D","coords":[[1.0000000000000002,-1.0000000000000004],[-3.0000000000000004,3.000000000000001]],"matrix":[[1.0000000000000002,1.0000000000000002,2.220446049250313e-16],[1.0000000000000004,1.0000000000000002,-2.220446049250313e-16],[0.125,-0.125,1]]},
{"id":"fma-cancellation-3D","coords":[[1.0000000000000002,-1.0000000000000004,0.3333333333333333],[-3.0000000000000004,3.000000000000001,-0.6666666666666666]],"matrix":[[1.0000000000000002,1.0000000000000002,-0.6666666666666666,0.2222222222222222],[1.0000000000000004,1.0000000000000002,0.3333333333333333,-0.1111111111111111],[0.6666666666666666,-0.3333333333333333,1.0000000000000002,-0.7777777777777778],[0.125,-0.125,0.0625,1]]},
{"id":"fma-cancellation-4D","coords":[[1.0000000000000002,-1.0000000000000004,0.3333333333333333,-0.6666666666666666],[-3.0000000000000004,3.000000000000001,-0.6666666666666666,0.3333333333333333]],"matrix":[[1.0000000000000002,1.0000000000000002,-0.6666666666666666,0.3333333333333333,0.4444444444444444],[1.0000000000000004,1.0000000000000002,0.3333333333333333,-0.6666666666666666,-0.5555555555555556],[0.6666666666666666,-0.3333333333333333,1.0000000000000002,0.7777777777777778,-0.8888888888888888],[-0.3333333333333333,0.6666666666666666,-0.7777777777777778,1.0000000000000004,0.9999999999999999],[0.125,-0.125,0.0625,-0.0625,1]]},
{"id":"zero-projective-denominator","coords":[[0.123456789,-0.34567891],[1234.567,2345.678]],"matrix":[[0.9987623,-0.03456789,12.3456],[0.04567891,1.0012345,-7.6543],[0,0,0]]},
{"id":"negative-zero-inputs","coords":[[-0.0,0.0],[0.0,-0.0]],"matrix":[[-0.0,0.0,-0.0],[0.0,-0.0,-0.0],[0.0,0.0,-0.0]]}
]}
*/
// MATRIX_COHORTS_END
#include "skimagexx/matrix_transform.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
using namespace skimagexx;
namespace {
std::size_t checks=0;
void check(bool yes) { ++checks;if(!yes)throw std::runtime_error("matrix native assertion "+std::to_string(checks)); }
MatrixReadView read(const std::vector<double>& x,std::size_t r,std::size_t c) {
    return {x.data(),r,c,std::ptrdiff_t(c*8),8,x.data(),x.size()*8};
}
void controls() {
    check(get_num_threads()==1);
    std::vector<double> p={1024.25,512.5,4095.75,2047.25},m={1,0,320,0,1,-240,0,0,1},o(4,-123.);
    auto pv=read(p,2,2),mv=read(m,3,3);
    MatrixWriteView ov{o.data(),2,2,o.size()*8};
    const auto run=[&](MatrixArithmeticProfile mode=MatrixArithmeticProfile::haswell_gemm_fma){return matrix_transform_into(pv,mv,ov,mode);};
    const auto reject=[&](MatrixMapStatus reason,MatrixArithmeticProfile mode=MatrixArithmeticProfile::haswell_gemm_fma){
        std::fill(o.begin(),o.end(),-123.);check(run(mode)==reason);check(std::all_of(o.begin(),o.end(),[](auto x){return x==-123.;}));
    };
    reject(MatrixMapStatus::unsupported_profile,MatrixArithmeticProfile::unsupported);
    pv.columns=1;reject(MatrixMapStatus::unsupported_shape);pv=read(p,2,2);
    pv.allocation_bytes=8;reject(MatrixMapStatus::invalid_view);pv=read(p,2,2);
    pv.row_stride=std::numeric_limits<std::ptrdiff_t>::min();reject(MatrixMapStatus::invalid_view);pv=read(p,2,2);
    ov.allocation_bytes=8;reject(MatrixMapStatus::invalid_view);ov.allocation_bytes=o.size()*8;
    ov.data=p.data();check(run()==MatrixMapStatus::overlap);check(p[0]==1024.25);ov.data=o.data();
    p.back()=std::numeric_limits<double>::infinity();reject(MatrixMapStatus::nonfinite_or_range);p.back()=2047.25;
    p.back()=0x1p-201;reject(MatrixMapStatus::nonfinite_or_range);p.back()=2047.25;
    // A late-row cancellation below the admitted divisor floor must reject
    // before overwriting even the first ordinary row of output.
    p[2]=p[3]=0x1p-200;m[6]=0x1p-200;m[7]=-std::nextafter(0x1p-200,1.);m[8]=0.;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    reject(MatrixMapStatus::unsafe_divisor);check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    std::feclearexcept(FE_ALL_EXCEPT);
    p[2]=4095.75;p[3]=2047.25;m[6]=0.;m[7]=0.;m[8]=1.;
    const auto real_output=ov.data;ov.data=reinterpret_cast<void*>(UINTPTR_MAX-7);
    reject(MatrixMapStatus::invalid_view);ov.data=real_output;
    const auto real_allocation=pv.allocation;pv.allocation=reinterpret_cast<void*>(UINTPTR_MAX-7);
    reject(MatrixMapStatus::invalid_view);pv.allocation=real_allocation;
    ov.data=m.data()+1;check(run()==MatrixMapStatus::overlap);check(m[1]==0.);ov.data=real_output;
    auto round=std::fegetround();std::fesetround(FE_DOWNWARD);reject(MatrixMapStatus::unsafe_fp_environment);std::fesetround(round);
#if defined(__x86_64__) || defined(_M_X64)
    const auto csr=_mm_getcsr();_mm_setcsr(csr|(1u<<15));reject(MatrixMapStatus::unsafe_fp_environment);_mm_setcsr(csr);
#endif
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    check(run()==MatrixMapStatus::ok);check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);std::feclearexcept(FE_ALL_EXCEPT);
    check(o==std::vector<double>({1344.25,272.5,4415.75,1807.25}));
    // Parallel workload crosses the real existing execution threshold.
    p.resize(262144);for(std::size_t i=0;i<p.size();++i)p[i]=double(i%4096)*0.125;
    o.resize(p.size());pv=read(p,p.size()/2,2);ov={o.data(),p.size()/2,2,o.size()*8};
    check(run()==MatrixMapStatus::ok);const auto serial=o;
#ifdef _OPENMP
    set_num_threads(4);check(get_num_threads()==4);check(run()==MatrixMapStatus::ok);
    check(std::memcmp(serial.data(),o.data(),o.size()*8)==0);set_num_threads(1);
#else
    bool rejected=false;try {set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected);check(get_num_threads()==1);
#endif
    // Opposite row signs, reverse and broadcast inputs preserve exact math.
    pv.data=p.data()+p.size()-2;pv.row_stride=-16;check(run()==MatrixMapStatus::ok);
    pv=read(p,p.size()/2,2);pv.row_stride=0;check(run()==MatrixMapStatus::ok);
    // Nontrivial FMA and cancellation across all D=2,3,4 and the true
    // 1/4-worker execution threshold. Each row keeps one exact dot tree.
    for(std::size_t d=2;d<=4;++d) {
        const std::size_t n=131073;
        p.resize(n*d);o.resize(n*d);m.resize((d+1)*(d+1));
        for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<d;++j)
            p[i*d+j]=(double((i+j)%4096)+0.123456789)*(j%2 ? -1.:1.);
        for(std::size_t i=0;i<=d;++i)for(std::size_t j=0;j<=d;++j)
            m[i*(d+1)+j]=i==d ? (j==d ? 1.:0.) : ((i+j)%2 ? -0.3333333333333333:0.6666666666666666);
        pv=read(p,n,d);mv=read(m,d+1,d+1);ov={o.data(),n,d,o.size()*8};
        check(run()==MatrixMapStatus::ok);const auto reference=o;
#ifdef _OPENMP
        set_num_threads(4);check(run()==MatrixMapStatus::ok);
        check(std::memcmp(reference.data(),o.data(),o.size()*8)==0);set_num_threads(1);
#endif
        // Compare reverse-strided reads to physically reversed C input.
        pv.data=p.data()+(n-1)*d;pv.row_stride=-std::ptrdiff_t(d*8);
        check(run()==MatrixMapStatus::ok);
        for(std::size_t i=0;i<n;++i)check(std::memcmp(o.data()+i*d,reference.data()+(n-1-i)*d,d*8)==0);
    }
    check(get_num_threads()==1);
}
void stream(int workers) {
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
    set_num_threads(workers);
    while(true) {
        // Fixed little-endian x86 private oracle protocol. No Python binding.
        std::uint64_t h[11];std::cin.read(reinterpret_cast<char*>(h),sizeof(h));
        if(std::cin.gcount()==0)return;
        if(!std::cin || h[1]>1000000 || h[2]>4 || h[6]>100000000 || h[10]>1000000)
            throw std::runtime_error("invalid oracle packet");
        std::vector<unsigned char> p(h[6]),m(h[10]);
        std::cin.read(reinterpret_cast<char*>(p.data()),p.size());std::cin.read(reinterpret_cast<char*>(m.data()),m.size());
        if(!std::cin || h[3]>=p.size() || h[7]>=m.size())throw std::runtime_error("truncated oracle packet");
        std::vector<double> o(h[1]*h[2],-123.);
        MatrixReadView pv{p.data()+h[3],std::size_t(h[1]),std::size_t(h[2]),std::ptrdiff_t(h[4]),std::ptrdiff_t(h[5]),p.data(),p.size()};
        MatrixReadView mv{m.data()+h[7],std::size_t(h[2]+1),std::size_t(h[2]+1),std::ptrdiff_t(h[8]),std::ptrdiff_t(h[9]),m.data(),m.size()};
        const auto status=matrix_transform_into(pv,mv,{o.data(),std::size_t(h[1]),std::size_t(h[2]),o.size()*8},MatrixArithmeticProfile(h[0]));
        const std::uint32_t result=std::uint32_t(status);std::cout.write(reinterpret_cast<const char*>(&result),4);
        std::cout.write(reinterpret_cast<const char*>(o.data()),o.size()*8);
    }
}
}
int main(int argc,char**argv) {
    try {
        if(argc>=2 && std::string(argv[1])=="--stream") {stream(argc==3?std::stoi(argv[2]):1);return 0;}
        controls();std::cout<<"matrix native checks "<<checks<<" PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
