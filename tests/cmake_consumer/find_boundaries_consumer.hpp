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
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace skimagexx_test {
// Independent clamped-neighbor controls for the F/F-to-C/C metadata route.
// The SDK operation owns the computation; the test never composes its stages.
template<class T>inline std::size_t find_boundaries_f_type(skimagexx::PixelType type,bool boolean,int workers) {
    using namespace skimagexx;using namespace skimagexx::detail;
    std::size_t checks=0;auto check=[&](bool v){++checks;if(!v)throw std::runtime_error("F-row boundary control failed");};
    for(auto shape:{std::vector<std::size_t>{1,9},{2,9},{3,3},{5,7},{7,5},{17,33},{33,17},{513,517}}){
        if(shape[0]>100 && sizeof(T)!=2)continue;
        const auto h=shape[0],w=shape[1],n=h*w;std::vector<T> input(n+16,T(0));auto* data=input.data()+8;
        for(std::size_t x=0;x<w;++x)for(std::size_t y=0;y<h;++y){
            const auto code=(y/3+2*x/5)%5;
            data[x*h+y]=boolean?T(code?code==1?2:255:0):code==4?std::numeric_limits<T>::max():T(code);
        }
        const auto before=input;ComponentsView view{data,type,false,false,shape,{std::ptrdiff_t(sizeof(T)),std::ptrdiff_t(h*sizeof(T))}};
        auto at=[&](std::size_t y,std::size_t x){T v=data[x*h+y];return boolean?T(v!=0):v;};
        for(auto mode:{FindBoundaryMode::thick,FindBoundaryMode::inner})for(unsigned connectivity:{1u,2u})
        for(int bg:{0,2,-1,99}){
            if(h>100 && (mode!=FindBoundaryMode::inner || connectivity!=2 || bg!=0))continue;
            FindBoundaryParameters parameters{mode,connectivity,std::uint64_t(bg),bg!=99};
            std::vector<std::uint8_t> expected(n);
            for(std::size_t x=0;x<w;++x)for(std::size_t y=0;y<h;++y){
                T low=std::numeric_limits<T>::max(),high=std::numeric_limits<T>::lowest();
                for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy){
                    if((dx!=0)+(dy!=0)>int(connectivity))continue;
                    auto yy=dy<0?(y?y-1:0):dy>0?std::min(y+1,h-1):y;
                    auto xx=dx<0?(x?x-1:0):dx>0?std::min(x+1,w-1):x;
                    auto value=at(yy,xx);low=std::min(low,value);high=std::max(high,value);
                }
                expected[x*h+y]=(low!=high)&&(mode!=FindBoundaryMode::inner || !parameters.background_representable || std::uint64_t(at(y,x))!=parameters.background_bits);
            }
            for(int budget:{0,workers}){
                std::vector<std::uint8_t> out(n+16,0xa7);const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();
                check(find_boundaries_into(view,parameters,out.data()+8,{1,std::ptrdiff_t(h)},budget));
                check(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==rounding);
                for(std::size_t i=0;i<n;++i)check(out[i+8]==expected[i]);
                for(std::size_t i=0;i<8;++i)check(out[i]==0xa7&&out[n+8+i]==0xa7);
                check(input==before);check(get_num_threads()==workers);
            }
        }
    }
    return checks;
}
inline std::size_t find_boundaries_f_rows(int workers) {
    using namespace skimagexx;
    const auto old=get_num_threads();struct Restore{int n;~Restore(){set_num_threads(n);}}restore{old};set_num_threads(workers);
    return find_boundaries_f_type<std::uint8_t>(PixelType::boolean,true,workers)
        +find_boundaries_f_type<std::uint8_t>(PixelType::uint8,false,workers)+find_boundaries_f_type<std::int8_t>(PixelType::int8,false,workers)
        +find_boundaries_f_type<std::uint16_t>(PixelType::uint16,false,workers)+find_boundaries_f_type<std::int16_t>(PixelType::int16,false,workers)
        +find_boundaries_f_type<std::uint32_t>(PixelType::uint32,false,workers)+find_boundaries_f_type<std::int32_t>(PixelType::int32,false,workers);
}
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
    // Exercise the signed/F traversal through the installed operation, including
    // its background mask and full-connectivity outer formula. Logical checker
    // labels make the expected mask independent of physical axis ordering.
    std::vector<std::uint16_t> f_labels(height*width);
    for(std::size_t y=0;y<height;++y)for(std::size_t x=0;x<width;++x)f_labels[x*height+y]=std::uint16_t(2+((x+y)&1));
    const auto saved_f=f_labels;
    ComponentsView f_view{f_labels.data()+height*(width-1),PixelType::uint16,false,false,
        {height,width},{2,-std::ptrdiff_t(2*height)}};
    for(auto mode:{FindBoundaryMode::inner,FindBoundaryMode::outer}){
        parameters={mode,2,2,true};
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();
        check(find_boundaries_into(f_view,parameters,out.data(),{std::ptrdiff_t(width),1},0));
        check(std::fetestexcept(FE_ALL_EXCEPT)==flags && std::fegetround()==rounding);
        for(std::size_t y=0;y<height;++y)for(std::size_t x=0;x<width;++x){
            const auto v=f_labels[(width-1-x)*height+y];
            check(out[y*width+x]==(mode==FindBoundaryMode::inner?v!=2:v==2));
        }
        check(f_labels==saved_f);check(get_num_threads()==workers);
    }
    std::vector<std::size_t> unchanged{99};view.shape={0,2};check(!find_boundaries_shape(view,parameters.mode,unchanged));check(unchanged==std::vector<std::size_t>{99});
    controls+=find_boundaries_f_rows(workers);
    return controls;
}
}
