#pragma once
#include <skimagexx/regular_grid.hpp>
#include <skimagexx/image.hpp>
#include "../fixtures/regular_grid_originals_v1.hpp"
#include <algorithm>
#include <cfenv>
#include <cstring>
#include <stdexcept>
namespace skimagexx_test {
inline std::size_t regular_grid_consumer(int workers){
    using namespace skimagexx;std::size_t checks=0;auto check=[&](bool value){++checks;if(!value)throw std::runtime_error("regular_grid installed SDK control failed");};
    const auto old=get_num_threads();struct Restore{int n;~Restore(){set_num_threads(n);}}restore{old};set_num_threads(workers);
    for(const auto& row:regular_grid_originals::rows()){
        const auto low=std::min<std::ptrdiff_t>(0,row.count ? std::ptrdiff_t(row.count-1)*row.stride : 0),high=std::max<std::ptrdiff_t>(0,row.count ? std::ptrdiff_t(row.count-1)*row.stride : 0);
        std::vector<std::uint8_t> storage(std::size_t(high-low)+row.bytes+16,0xad);auto* base=storage.data()+8-low;
        for(std::size_t i=0;i<row.count;++i)std::memcpy(base+std::ptrdiff_t(i)*row.stride,row.input.data()+i*row.bytes,row.bytes);
        const auto before=storage;ComponentsView view{base,row.type,false,row.swap,{row.count},{row.stride}};
        // Match retained capture's nearest/inexact baseline; do not hide new
        // flags from the operation. Each case restores the caller environment.
        struct FP{std::fenv_t saved;FP(){std::fegetenv(&saved);}~FP(){std::fesetenv(&saved);}}fp;
        check(std::fesetround(FE_TONEAREST)==0);check(std::feclearexcept(FE_ALL_EXCEPT)==0);check(std::feraiseexcept(FE_INEXACT)==0);
        bool division_error=false;std::vector<RegularGridSlice> result;
        try{result=regular_grid(view,row.points,8);}catch(const RegularGridError& e){check(e.kind==RegularGridErrorKind::division_zero);check(std::string(e.what())=="float division by zero");division_error=true;}
        check(division_error==row.division_error);check(result.size()==row.expected.size());
        for(std::size_t i=0;i<result.size();++i){check(result[i].kind==row.expected[i].kind);if(result[i].kind==RegularGridSliceKind::strided){check(result[i].start==row.expected[i].start);check(result[i].step==row.expected[i].step);}}
        check(storage==before);check(get_num_threads()==workers);
        check(std::fetestexcept(FE_ALL_EXCEPT)==FE_INEXACT);check(std::fegetround()==FE_TONEAREST);
    }
    // Meaningful ranks/shape sizes, independent of reference recipe bounds.
    const std::int64_t shape[]{4093,13,997};ComponentsView view{shape,PixelType::int64,false,false,{3},{8}};
    auto ordinary=regular_grid(view,6000);check(ordinary.size()==3);check(ordinary[1].step==13);
    const std::int64_t ties[]{25};view={ties,PixelType::int64,false,false,{1},{8}};
    auto result=regular_grid(view,10);check(result[0].start==1 && result[0].step==2);
    const std::int64_t ties_up[]{35};view.data=ties_up;result=regular_grid(view,10);check(result[0].start==1 && result[0].step==4);
    std::vector<RegularGridSlice> compatibility(3);const std::int64_t dimensions[]{3,20,40};check(detail::regular_grid_slices_i64(dimensions,3,8,compatibility.data()));check(compatibility[0].start==1 && compatibility[0].step==3);
    const std::int16_t product_width[]{20000,20000,10};view={product_width,PixelType::int16,false,false,{3},{2}};
    auto narrow=regular_grid(view,1,4),wide=regular_grid(view,1,8);
    check(std::all_of(narrow.begin(),narrow.end(),[](const auto& s){return s.kind==RegularGridSliceKind::all;}));
    check(std::any_of(wide.begin(),wide.end(),[](const auto& s){return s.kind==RegularGridSliceKind::strided;}));
    check(get_num_threads()==workers);return checks;
}
}
