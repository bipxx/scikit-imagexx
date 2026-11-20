#include <skimagexx/order_angles_golden_ratio.hpp>
#include <skimagexx/image.hpp>
#include "order_angles_golden_ratio_fenv.hpp"
#include <cassert>
#include <cfenv>
#include <iostream>
#include <limits>
using namespace skimagexx;
int main(int argc,char** argv) {
    assert(argc==2);GoldenFPEnvelope environment(argv[1]);
    assert(get_num_threads()==1);double input[]={90,0,30,60};ComponentsView v{input,PixelType::float64,false,false,{4},{8}};
    GoldenAngleOrder lazy(v);assert(lazy.worker_budget()==0);set_num_threads(4);const auto first=lazy.begin();assert(first==1&&lazy.worker_budget()==4);set_num_threads(1);
    while(lazy.resume()){(void)lazy.next(v);assert(lazy.worker_budget()==4);}
    bool rejected=false;try{GoldenAngleOrder bad({input,PixelType::float64,false,false,{std::size_t(PTRDIFF_MAX)},{8}});}catch(const std::overflow_error&){rejected=true;}assert(rejected);
    GoldenAngleOrder live(v);(void)live.begin();assert(live.resume());auto changed=v;changed.strides={16};rejected=false;try{live.next(changed);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    // Source NumPy policy boundary clears FOUR exception classes before real
    // arithmetic, preserves inexact; firstsort/increment do not clear them.
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID|FE_INEXACT);
    GoldenAngleOrder fp(v);(void)fp.begin();assert(std::fetestexcept(FE_INVALID));assert(fp.resume());assert(std::fetestexcept(FE_INVALID));
    (void)fp.next(v);assert(!std::fetestexcept(FE_INVALID));assert(std::fetestexcept(FE_INEXACT));
    std::cout<<"NATIVE lazy_worker_snapshot=PASS metadata_frontiers=PASS source_scalar_FP_boundary=PASS default="<<get_num_threads()<<"\n";environment.finish();
}
