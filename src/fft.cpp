#include "skimagexx/image.hpp"
#define POCKETFFT_NAMESPACE pocketfft_scalar
#define POCKETFFT_NO_VECTORS 1
#define FFT_LEAF fft_scalar
#include "fft_impl.hpp"

namespace skimagexx::detail {
#ifdef SKIMAGEXX_AVX2
namespace fft_avx2 {
void r2c_full(const std::vector<std::size_t>&,const std::vector<std::size_t>&,const float*,std::complex<float>*,bool,int);
void r2c_full(const std::vector<std::size_t>&,const std::vector<std::size_t>&,const double*,std::complex<double>*,bool,int);
void r2c(const std::vector<std::size_t>&,const std::vector<std::size_t>&,const float*,std::complex<float>*,int);
void c2r(const std::vector<std::size_t>&,const std::vector<std::size_t>&,const std::complex<float>*,float*,int);
void c2c(const std::vector<std::size_t>&,const std::vector<std::size_t>&,const std::complex<float>*,std::complex<float>*,bool,int);
void r2c(const std::vector<std::size_t>&,const std::vector<std::size_t>&,const double*,std::complex<double>*,int);
void c2r(const std::vector<std::size_t>&,const std::vector<std::size_t>&,const std::complex<double>*,double*,int);
void c2c(const std::vector<std::size_t>&,const std::vector<std::size_t>&,const std::complex<double>*,std::complex<double>*,bool,int);
}
#endif
void fft_r2c(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const float* input,std::complex<float>* output,int workers) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) { fft_avx2::r2c(shape,axes,input,output,workers); return; }
#endif
    fft_scalar::r2c(shape,axes,input,output,workers);
}
void fft_c2r(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const std::complex<float>* input,float* output,int workers) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) { fft_avx2::c2r(shape,axes,input,output,workers); return; }
#endif
    fft_scalar::c2r(shape,axes,input,output,workers);
}
void fft_c2c(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const std::complex<float>* input,std::complex<float>* output,bool forward,int workers) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) { fft_avx2::c2c(shape,axes,input,output,forward,workers); return; }
#endif
    fft_scalar::c2c(shape,axes,input,output,forward,workers);
}
void fft_r2c(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const double* input,std::complex<double>* output,int workers) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) { fft_avx2::r2c(shape,axes,input,output,workers); return; }
#endif
    fft_scalar::r2c(shape,axes,input,output,workers);
}
void fft_c2r(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const std::complex<double>* input,double* output,int workers) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) { fft_avx2::c2r(shape,axes,input,output,workers); return; }
#endif
    fft_scalar::c2r(shape,axes,input,output,workers);
}
void fft_c2c(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const std::complex<double>* input,std::complex<double>* output,bool forward,int workers) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) { fft_avx2::c2c(shape,axes,input,output,forward,workers); return; }
#endif
    fft_scalar::c2c(shape,axes,input,output,forward,workers);
}
void fft_r2c_full(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const float* input,std::complex<float>* output,bool forward,int workers) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) { fft_avx2::r2c_full(shape,axes,input,output,forward,workers); return; }
#endif
    fft_scalar::r2c_full(shape,axes,input,output,forward,workers);
}
void fft_r2c_full(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const double* input,std::complex<double>* output,bool forward,int workers) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) { fft_avx2::r2c_full(shape,axes,input,output,forward,workers); return; }
#endif
    fft_scalar::r2c_full(shape,axes,input,output,forward,workers);
}
}
