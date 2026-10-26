#define POCKETFFT_NAMESPACE pocketfft_avx2
#define FFT_LEAF fft_avx2_impl
#include "fft_impl.hpp"
namespace skimagexx::detail::fft_avx2 {
void r2c(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const float* input,std::complex<float>* output,int workers) { fft_avx2_impl::r2c(shape,axes,input,output,workers); }
void c2r(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const std::complex<float>* input,float* output,int workers) { fft_avx2_impl::c2r(shape,axes,input,output,workers); }
void c2c(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const std::complex<float>* input,std::complex<float>* output,bool forward,int workers) { fft_avx2_impl::c2c(shape,axes,input,output,forward,workers); }
void r2c(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const double* input,std::complex<double>* output,int workers) { fft_avx2_impl::r2c(shape,axes,input,output,workers); }
void c2r(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const std::complex<double>* input,double* output,int workers) { fft_avx2_impl::c2r(shape,axes,input,output,workers); }
void c2c(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& axes,const std::complex<double>* input,std::complex<double>* output,bool forward,int workers) { fft_avx2_impl::c2c(shape,axes,input,output,forward,workers); }
}
