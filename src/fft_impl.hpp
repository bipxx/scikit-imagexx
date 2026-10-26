// Included once per scalar/AVX2 TU, with distinct header and leaf namespaces.
#include "fft_internal.hpp"
#include <exception>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#define SKIMAGEXX_POCKETFFT_OPENMP 1
#else
#define POCKETFFT_NO_MULTITHREADING 1
#endif
#define POCKETFFT_CACHE_SIZE 16
#define pocketfft POCKETFFT_NAMESPACE
#include "../third_party/pocketfft/pocketfft_openmp.h"
#undef pocketfft

namespace skimagexx::detail::FFT_LEAF {
struct Layout {
    std::vector<std::ptrdiff_t> real, complex, full_complex;
    std::size_t transform_size=1;
};
inline Layout layout(const std::vector<std::size_t>& shape,
                     const std::vector<std::size_t>& axes, std::size_t item, int workers) {
    if (shape.empty() || axes.empty()) throw std::invalid_argument("FFT shape/axes must be nonempty");
    if (workers<1 || workers>1024) throw std::invalid_argument("FFT worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if (workers!=1) throw std::invalid_argument("this build has no OpenMP support");
#endif
    std::vector<bool> used(shape.size());
    Layout l;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(2*item);
    std::size_t count=1;
    for (auto size:shape) {
        if (!size || count>limit/size) throw std::invalid_argument("FFT shape extent overflow or zero");
        count*=size;
    }
    for (auto axis:axes) {
        if (axis>=shape.size() || used[axis]) throw std::invalid_argument("FFT axes invalid or repeated");
        used[axis]=true; l.transform_size*=shape[axis];
    }
    l.real.resize(shape.size()); l.complex.resize(shape.size()); l.full_complex.resize(shape.size());
    std::size_t rs=item, cs=2*item, fs=2*item;
    for (std::size_t i=shape.size(); i-- >0;) {
        l.real[i]=std::ptrdiff_t(rs); l.complex[i]=std::ptrdiff_t(cs); l.full_complex[i]=std::ptrdiff_t(fs);
        rs*=shape[i]; fs*=shape[i]; cs*=i==axes.back() ? shape[i]/2+1 : shape[i];
    }
    return l;
}
template<class T> void r2c(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
                          const T* input, std::complex<T>* output, int workers) {
    auto l=layout(shape,axes,sizeof(T),workers);
    if (!input || !output) throw std::invalid_argument("FFT null buffer");
    POCKETFFT_NAMESPACE::r2c(shape,l.real,l.complex,axes,true,input,output,T(1),std::size_t(workers));
}
template<class T> void c2r(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
                          const std::complex<T>* input, T* output, int workers) {
    auto l=layout(shape,axes,sizeof(T),workers);
    if (!input || !output) throw std::invalid_argument("FFT null buffer");
    const auto scale=T(1.L/static_cast<long double>(l.transform_size));
    POCKETFFT_NAMESPACE::c2r(shape,l.complex,l.real,axes,false,input,output,scale,std::size_t(workers));
}
template<class T> void c2c(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
                          const std::complex<T>* input, std::complex<T>* output, bool forward, int workers) {
    auto l=layout(shape,axes,sizeof(T),workers);
    if (!input || !output) throw std::invalid_argument("FFT null buffer");
    const auto scale=forward ? T(1) : T(1.L/static_cast<long double>(l.transform_size));
    POCKETFFT_NAMESPACE::c2c(shape,l.full_complex,l.full_complex,axes,forward,input,output,scale,std::size_t(workers));
}
template<class T> void r2c_full(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
                               const T* input, std::complex<T>* output, bool forward, int workers) {
    auto l=layout(shape,axes,sizeof(T),workers);
    if (!input || !output) throw std::invalid_argument("FFT null buffer");
    const auto scale=forward ? T(1) : T(1.L/static_cast<long double>(l.transform_size));
    POCKETFFT_NAMESPACE::r2c(shape,l.real,l.full_complex,axes,forward,input,output,scale,std::size_t(workers));
    // SciPy c2c_sym_internal: preserve its complete serial iteration order,
    // including self-conjugate/boundary planes and every transformed axis.
    POCKETFFT_NAMESPACE::detail::ndarr<std::complex<T>> values(output,shape,l.full_complex);
    POCKETFFT_NAMESPACE::detail::rev_iter iter(values,axes);
    while (iter.remaining()>0) {
        auto value=values[iter.ofs()];
        values[iter.rev_ofs()]=std::conj(value);
        iter.advance();
    }
}
}
