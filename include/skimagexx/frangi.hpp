#pragma once
#include <cstddef>
namespace skimagexx {
struct FrangiEigenProvider {
 bool (*evaluate)(const void* const*,void*,std::size_t,std::size_t,bool,void*)=nullptr;
 void* context=nullptr;
};
struct FrangiExpProvider {
 // Separate input/output buffers. Host-specific SIMD double-exp rounding.
 void (*evaluate)(const double*,double*,std::size_t,void*)=nullptr;
 void* context=nullptr;
};
struct FrangiFloatExpProvider {
 void (*evaluate)(const float*,float*,std::size_t,void*)=nullptr;
 void* context=nullptr;
};
struct FrangiParameters {
 double alpha_den=0.5,beta_den=0.5,gamma_den=0.;
 bool alpha_single=false,beta_single=false,gamma_single=false;
 bool automatic_gamma=true,gamma_compute_single=false;
 // Optional native math provider for application-specific LAPACK rounding.
 // Output is decreasing eigenvalue fields in the input field precision.
 // Null uses the standalone C++ Jacobi/formula solver.
 const FrangiEigenProvider* eigen_provider=nullptr;
 // Null uses the C++ standard library exponential.
 const FrangiExpProvider* exp_provider=nullptr;
 // Optional host float-exp rounding when the portable approximation differs.
 const FrangiFloatExpProvider* float_exp_provider=nullptr;
};
// Homogeneous dense/aligned upper-triangle Hessian fields in rank2/3. Output
// and previous (or null=zeros) match field precision, separate storage.
// Denominators are 2*parameter^2 with explicit division precision. Automatic
// gamma is initialized once from this scale's maximum Hessian norm; the
// resulting denominator and automatic_gamma=false are returned in parameters.
// False rejects unsafe arithmetic/nonfinite data; discard output and plan.
// Global worker cap defaults one. Exp uses field precision after division.
bool frangi_accumulate_into(const void* const*,const void*,void*,std::size_t,
                            std::size_t,bool single,FrangiParameters&);
// Ridge Hessian's final transformation: nonpositive response -> one. In place.
void ridge_hessian_finish_into(void*,std::size_t,bool single);
}
