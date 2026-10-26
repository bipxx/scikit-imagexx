#include "fft_internal.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef NDEBUG
#error FFT native assertions must remain enabled
#endif
#ifdef _OPENMP
#include <omp.h>
#define SKIMAGEXX_POCKETFFT_OPENMP 1
#else
#define POCKETFFT_NO_MULTITHREADING 1
#endif
#define POCKETFFT_NAMESPACE pocketfft_test
#define POCKETFFT_NO_VECTORS 1
#define pocketfft POCKETFFT_NAMESPACE
#include "../third_party/pocketfft/pocketfft_openmp.h"
#undef pocketfft

using namespace skimagexx;
using namespace skimagexx::detail;
using Shape=std::vector<std::size_t>;
static std::size_t checks=0;
void check(bool condition) { ++checks; if (!condition) throw std::runtime_error("FFT native control failed at "+std::to_string(checks)); }
std::size_t product(const Shape& shape) { std::size_t n=1; for(auto v:shape)n*=v; return n; }
Shape coord(std::size_t index,const Shape& shape) {
    Shape v(shape.size()); for(std::size_t i=shape.size();i-- >0;) { v[i]=index%shape[i];index/=shape[i]; } return v;
}
// Independent direct DFT: enumerate full coordinates, not FFT decomposition.
template<class T> std::vector<std::complex<long double>> dft(const std::vector<std::complex<T>>& in,const Shape& shape,const Shape& axes) {
    auto n=product(shape); std::vector<std::complex<long double>> out(n);
    std::vector<bool> selected(shape.size()); for(auto a:axes)selected[a]=true;
    const auto tau=2*std::acos(-1.L);
    for(std::size_t k=0;k<n;++k) {
        auto kc=coord(k,shape);
        for(std::size_t j=0;j<n;++j) {
            auto jc=coord(j,shape); bool same=true; long double phase=0;
            for(std::size_t a=0;a<shape.size();++a) {
                if(selected[a]) phase+=static_cast<long double>(jc[a]*kc[a])/shape[a];
                else same= same && jc[a]==kc[a];
            }
            if(same)out[k]+=std::complex<long double>(in[j])*std::complex<long double>(std::cos(tau*phase),-std::sin(tau*phase));
        }
    } return out;
}
template<class T> void small(const Shape& shape,const Shape& axes,int workers) {
    auto n=product(shape); std::vector<std::complex<T>> in(n),out(n),back(n);
    std::vector<T> real(n),roundtrip(n);
    for(std::size_t i=0;i<n;++i) { real[i]=T(int(i%11)-5)/T(8); in[i]={real[i],T(int(i%7)-3)/T(16)}; }
    auto expected=dft(in,shape,axes); auto old=in;
    const auto tolerance=std::numeric_limits<T>::epsilon()*n*64;
    fft_c2c(shape,axes,in.data(),out.data(),true,workers);
    check(in==old);
    for(std::size_t i=0;i<n;++i)check(std::abs(std::complex<long double>(out[i])-expected[i])<tolerance);
    fft_c2c(shape,axes,out.data(),back.data(),false,workers);
    for(std::size_t i=0;i<n;++i)check(std::abs(back[i]-in[i])<tolerance);
    for(std::size_t i=0;i<n;++i)in[i]={real[i],0};
    expected=dft(in,shape,axes);
    auto packed=shape; packed[axes.back()]=shape[axes.back()]/2+1;
    std::vector<std::complex<T>> spectrum(product(packed));
    fft_r2c(shape,axes,real.data(),spectrum.data(),workers);
    for(std::size_t k=0;k<spectrum.size();++k) {
        auto c=coord(k,packed); std::size_t full=0; for(std::size_t a=0;a<shape.size();++a)full=full*shape[a]+c[a];
        check(std::abs(std::complex<long double>(spectrum[k])-expected[full])<tolerance);
    }
    const auto preserved=spectrum;
    fft_c2r(shape,axes,spectrum.data(),roundtrip.data(),workers); check(spectrum==preserved);
    for(std::size_t i=0;i<n;++i)check(std::abs(roundtrip[i]-real[i])<tolerance);
}
template<class T> void large(int workers) {
    const Shape shape{64,64,65},axes{0,1,2}; const auto n=product(shape);
    std::vector<T> in(n),back(n); in[0]=1;
    std::vector<std::complex<T>> out(64*64*33);
    fft_r2c(shape,axes,in.data(),out.data(),workers);
    for(auto z:out)check(z==std::complex<T>(1,0));
    fft_c2r(shape,axes,out.data(),back.data(),workers);
    for(std::size_t i=0;i<n;++i)check(std::abs(back[i]-in[i])<std::numeric_limits<T>::epsilon()*32);
}
void invalid(int workers) {
    double in[4]={1,2,3,4}; std::complex<double> out[4]={{77,88},{77,88},{77,88},{77,88}};
    for(auto args:std::vector<std::pair<Shape,Shape>>{{{},{}},{{2,0},{0}},{{2,2},{2}},{{2,2},{0,0}},{{std::numeric_limits<std::size_t>::max(),2},{0}}}) {
        bool refused=false; try {fft_r2c(args.first,args.second,in,out,workers);}catch(const std::invalid_argument&){refused=true;}
        check(refused); for(auto z:out)check(z==std::complex<double>(77,88));
    }
    bool refused=false; try {fft_r2c({4},{0},in,out,0);}catch(const std::invalid_argument&){refused=true;} check(refused);
    refused=false; try {fft_r2c({4},{0},static_cast<const double*>(nullptr),out,workers);}catch(const std::invalid_argument&){refused=true;}check(refused);
}
void thread_hook(int workers) {
#ifdef _OPENMP
    std::atomic<int> calls{0}; std::atomic<bool> valid{true};
    auto old_id=pocketfft_test::detail::threading::thread_id();
    auto old_count=pocketfft_test::detail::threading::num_threads();
    pocketfft_test::detail::threading::thread_map(workers,[&] {
        auto id=pocketfft_test::detail::threading::thread_id(),team=pocketfft_test::detail::threading::num_threads();
        if(team>std::size_t(workers)||id>=team)valid=false;
        ++calls;
    });
    check(valid && calls>=1 && calls<=workers);
    bool caught=false;
    try {pocketfft_test::detail::threading::thread_map(workers,[] {throw std::runtime_error("test worker allocation failure");});}
    catch(const std::runtime_error& e) {caught=std::string(e.what())=="test worker allocation failure";}
    check(caught); check(pocketfft_test::detail::threading::thread_id()==old_id);check(pocketfft_test::detail::threading::num_threads()==old_count);
#else
    (void)workers;
#endif
}
int main() {
    check(get_num_threads()==1); auto rounding=std::fegetround();
    for(int workers:{1,4}) {
#ifndef _OPENMP
        if(workers==4) {bool refused=false;try{set_num_threads(4);}catch(const std::invalid_argument&){refused=true;}check(refused);break;}
#endif
        set_num_threads(workers);
        for(auto shape:std::vector<Shape>{{1},{2},{5},{7},{8},{3,4},{3,5},{2,3,4}}) {
            Shape axes;for(std::size_t a=0;a<shape.size();++a)axes.push_back(a);
            small<float>(shape,axes,workers);small<double>(shape,axes,workers);
        }
        for(auto axes:std::vector<Shape>{{0},{1},{2},{2,0},{1,0}}) {small<float>({2,3,5},axes,workers);small<double>({2,3,5},axes,workers);}
        invalid(workers);thread_hook(workers);large<float>(workers);large<double>(workers);
        check(std::fegetround()==rounding);
    }
    set_num_threads(1); std::cout<<"FFT controls "<<checks<<" PASS avx2="<<has_avx2()<<"\n";
}
