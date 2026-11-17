#pragma once
#include <skimagexx/finite_radon.hpp>
#include <skimagexx/image.hpp>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "../fixtures/finite_radon_saved_v2.hpp"
#include "../fixtures/finite_radon_exceptional_saved_v1.hpp"
inline void finite_radon_saved_consumer(){
    using namespace skimagexx;unsigned count=0;
    for(bool modern:{false,true})for(auto row:finite_radon_saved(modern))for(int workers:{1,4}){
        set_num_threads(workers);std::vector<std::ptrdiff_t> strides;
        std::size_t width=row.half?2:pixel_size(row.type)*(row.complex?2:1);strides={row.shape[1]*std::ptrdiff_t(width),std::ptrdiff_t(width)};
        FiniteRadonInput in{{row.input.data(),row.shape,strides,width},row.type,row.swapped,row.half,row.complex};auto before=row.input;
        if(row.inverse){auto out=ifrt2(in,{64,row.weak,row.signed_sum});assert(out.captured_workers==workers&&out.values.size()*8==row.gold.size());if(!row.gold.empty())assert(std::memcmp(out.values.data(),row.gold.data(),row.gold.size())==0);}
        else{auto out=frt2(in);assert(out.captured_workers==workers&&out.values.size()*4==row.gold.size());if(!row.gold.empty())assert(std::memcmp(out.values.data(),row.gold.data(),row.gold.size())==0);}
        assert(row.input==before);++count;
    }
    assert(count==244);set_num_threads(1);std::cout<<"PASS admitted61returns/profile completeSDK logical words x workers1/4=244\n";
}
inline void finite_radon_exceptional_consumer(){
    using namespace skimagexx;unsigned returns=0,errors=0;
    for(bool modern:{false,true})for(auto row:finite_radon_exceptional_saved(modern))for(int workers:{1,4}){
        const std::ptrdiff_t n=5,rows=n+(row.inverse?1:0);
        FiniteRadonInput input{{row.input.data(),{rows,n},{n*8,8},8},PixelType::float64,false};
        const auto before=row.input;std::vector<unsigned> notices;
        auto notice=[](void* context,unsigned event){static_cast<std::vector<unsigned>*>(context)->push_back(event);};
        set_num_threads(workers);
        if(row.inverse){bool failed=false;try{(void)ifrt2(input,{},0,notice,&notices);}catch(const FiniteRadonCastingError&){failed=true;}assert(failed);++errors;}
        else{auto out=frt2(input,0,notice,&notices);assert(out.captured_workers==workers&&out.values.size()*4==row.gold.size());assert(std::memcmp(out.values.data(),row.gold.data(),row.gold.size())==0);++returns;}
        assert(row.input==before&&notices.size()==row.notices);for(auto event:notices)assert(event==2);
    }
    // Native-only cast-policy invariant: fractional signed32 endpoint truncates
    // validly (inexact is not invalid). This is not another original witness.
    double edge[16]={-2147483648.5};std::vector<unsigned> policy;
    auto check=[](void* p,unsigned e){static_cast<std::vector<unsigned>*>(p)->push_back(e);};
    (void)frt2({{edge,{4,4},{32,8},8},PixelType::float64,false},1,check,&policy);
    assert(policy.empty());
    assert(returns==8&&errors==8);set_num_threads(1);std::cout<<"PASS exceptional4returns xworkers1/4=8 plus8 complete inverse typederrors/notices\n";
}
inline void finite_radon_consumer(){
    using namespace skimagexx;
    assert(get_num_threads()==1);
    finite_radon_saved_consumer();
    finite_radon_exceptional_consumer();
    // Independent direct coordinate definition, not saved original gold or a
    // copied rolling implementation. Small prime/nonprime and wrap domains.
    for(std::size_t n:{0,1,2,3,4,5,17,65}){
        std::vector<std::uint32_t> image(n*n);for(std::size_t k=0;k<image.size();++k)image[k]=k%3?std::uint32_t(k*12345):0xffffffffu;
        auto before=image;FiniteRadonInput in{{image.data(),{std::ptrdiff_t(n),std::ptrdiff_t(n)},{std::ptrdiff_t(n*4),4},4},PixelType::uint32,false};
        for(int workers:{1,4}){set_num_threads(workers);auto result=frt2(in);assert(result.captured_workers==workers&&result.rows==n+1&&result.columns==n);
            for(std::size_t m=0;m<n;++m)for(std::size_t c=0;c<n;++c){std::uint32_t value=0;for(std::size_t r=0;r<n;++r)value+=image[r*n+(c+m*r)%n];assert(result.values[m*n+c]==value);}
            for(std::size_t r=0;r<n;++r){std::uint32_t sum=0;for(std::size_t c=0;c<n;++c)sum+=image[r*n+c];assert(result.values[n*n+r]==sum);}
            assert(image==before);
            if(n){FiniteRadonInput inv{{result.values.data(),{std::ptrdiff_t(n+1),std::ptrdiff_t(n)},{std::ptrdiff_t(n*4),4},4},PixelType::uint32,false};
                for(bool weak:{false,true}){auto out=ifrt2(inv,{64,weak,false});assert(out.captured_workers==workers&&out.values.size()==n*n);std::uint64_t total=0;for(std::size_t c=0;c<n;++c)total+=result.values[c];
                    for(std::size_t m=0;m<n;++m)for(std::size_t c=0;c<n;++c){std::uint32_t value=result.values[n*n+m];for(std::size_t r=0;r<n;++r)value+=result.values[r*n+(c+n-(m*r)%n)%n];const double expected=weak&&total<=0xffffffffu?double(std::uint32_t(value-std::uint32_t(total)))/double(n):double(std::uint64_t(value)-total)/double(n);assert(out.values[m*n+c]==expected);}
                }
            }
        }
    }
    // Prime positive-domain round trip, ownership survives input destruction.
    InverseFiniteRadonTransform owned;
    {std::vector<std::uint8_t> input(25);for(std::size_t k=0;k<25;++k)input[k]=std::uint8_t(k);auto f=frt2({{input.data(),{5,5},{5,1},1},PixelType::uint8,false});owned=ifrt2({{f.values.data(),{6,5},{20,4},4},PixelType::uint32,false});}
    for(std::size_t k=0;k<25;++k)assert(owned.values[k]==double(k));
    std::int8_t negative[4]={-1,-2,-3,-4};auto signed_result=frt2({{negative+2,{2,2},{-2,1},1},PixelType::int8,false});assert(signed_result.values[0]==std::uint32_t(-4)&&signed_result.values[1]==std::uint32_t(-6));
    unsigned char big[8]={0,1,0,2,0,3,0,4};auto swapped=frt2({{big,{2,2},{4,2},2},PixelType::uint16,true});assert(swapped.values[0]==4&&swapped.values[1]==6);
    std::uint32_t witness[6]={10,0,0,0,0,0};FiniteRadonInput inv{{witness,{3,2},{8,4},4},PixelType::uint32,false};auto weak=ifrt2(inv,{64,true,false}),strong=ifrt2(inv,{64,false,false});assert(weak.values[1]==double(std::uint32_t(0u-10u))/2);assert(strong.values[1]==double(std::uint64_t(0)-10)/2);assert(weak.values[1]!=strong.values[1]);
    bool failed=false;try{(void)ifrt2({{nullptr,{1,0},{0,4},4},PixelType::uint32,false});}catch(const std::out_of_range&){failed=true;}assert(failed);
    // Complete inverse floating operation performs native prefix and ordered
    // notices before its source in-place casting failure, not a stage-only API.
    std::vector<float> complex(24,1);std::vector<unsigned> events;
    auto callback=[](void* p,unsigned event){static_cast<std::vector<unsigned>*>(p)->push_back(event);};failed=false;
    try{(void)ifrt2({{complex.data(),{4,3},{24,8},8},PixelType::float32,false,false,true},{},0,callback,&events);}catch(const FiniteRadonCastingError&){failed=true;}
    assert(failed&&events==std::vector<unsigned>({1,1,1}));
    set_num_threads(1);std::cout<<"PASS native coordinate/wrap/promotion/ownership/stride/SIMD/workers invariants; NOT original parity\n";
}
