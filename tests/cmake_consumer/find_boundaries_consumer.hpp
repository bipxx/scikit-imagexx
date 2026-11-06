#pragma once
#include <skimagexx/find_boundaries.hpp>
#include <skimagexx/image.hpp>
#include "../fixtures/find_boundaries_originals_v1.hpp"
#include <algorithm>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace skimagexx_test {
// New helper for the EXISTING component/segmentation installed-SDK family.
// Root integration appends this include/default call; this header adds no family.
inline std::size_t find_boundaries_consumer(int workers) {
    using namespace skimagexx;using namespace skimagexx::detail;
    std::size_t controls=0;auto check=[&](bool passed){++controls;if(!passed)throw std::runtime_error("installed find_boundaries SDK control failed");};
    const auto old=get_num_threads();
    struct Restore {int budget;~Restore(){set_num_threads(budget);}}restore{old};
    set_num_threads(workers);check(get_num_threads()==workers);
    auto product=[](const std::vector<std::size_t>& shape){return std::accumulate(shape.begin(),shape.end(),std::size_t{1},std::multiplies<std::size_t>());};
    std::size_t accepted=0,refused=0;
    for(const auto& row:find_boundaries_originals::rows()) {
        const auto elements=product(row.shape),item=row.input.size()/elements;
        std::ptrdiff_t low=0,high=0;
        for(std::size_t d=0;d<row.shape.size();++d){const auto span=std::ptrdiff_t(row.shape[d]-1)*row.strides[d];low+=std::min<std::ptrdiff_t>(0,span);high+=std::max<std::ptrdiff_t>(0,span);}
        std::vector<std::uint8_t> input(std::size_t(high-low)+item+16,0xad);auto* base=input.data()+8-low;
        for(std::size_t i=0;i<elements;++i){auto logical=i;std::ptrdiff_t offset=0;for(auto d=row.shape.size();d-->0;){offset+=std::ptrdiff_t(logical%row.shape[d])*row.strides[d];logical/=row.shape[d];}std::memcpy(base+offset,row.input.data()+i*item,item);}
        const auto before=input;ComponentsView view{base,row.type,false,false,row.shape,row.strides};std::vector<std::size_t> shape;
        check(find_boundaries_shape(view,row.parameters.mode,shape));check(product(shape)==row.output.size());
        std::vector<std::uint8_t> output(row.output.size()+16,0xad);
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();
        const bool admitted=find_boundaries_into(view,row.parameters,output.data()+8,row.output_strides,0);
        check(admitted==row.admitted);check(std::fetestexcept(FE_ALL_EXCEPT)==flags);check(std::fegetround()==rounding);check(get_num_threads()==workers);
        if(!admitted){++refused;check(std::all_of(output.begin(),output.end(),[](auto b){return b==0xad;}));continue;}
        ++accepted;
        for(std::size_t i=0;i<row.output.size();++i){auto logical=i;std::ptrdiff_t offset=0;for(auto d=shape.size();d-->0;){offset+=std::ptrdiff_t(logical%shape[d])*row.output_strides[d];logical/=shape[d];}check(output[8+offset]==row.output[i]);}
        check(std::all_of(output.begin(),output.begin()+8,[](auto b){return b==0xad;}));check(std::all_of(output.end()-8,output.end(),[](auto b){return b==0xad;}));check(input==before);
    }
    check(accepted==56);check(refused==2);
    // Actually crosses the existing parallel threshold on the SIMD row path.
    const std::size_t height=513,width=517;std::vector<std::uint8_t> labels(height*width),out(labels.size());
    for(std::size_t y=0;y<height;++y)for(std::size_t x=0;x<width;++x)labels[y*width+x]=((x+y)&1)?255:0;
    ComponentsView view{labels.data(),PixelType::boolean,false,false,{height,width},{std::ptrdiff_t(width),1}};
    FindBoundaryParameters parameters;
    check(find_boundaries_into(view,parameters,out.data(),{std::ptrdiff_t(width),1},0));
    for(auto value:out)check(value==1);
    check(get_num_threads()==workers);
    std::vector<std::size_t> unchanged{99};view.shape={0,2};check(!find_boundaries_shape(view,parameters.mode,unchanged));check(unchanged==std::vector<std::size_t>{99});
    return controls;
}
}
