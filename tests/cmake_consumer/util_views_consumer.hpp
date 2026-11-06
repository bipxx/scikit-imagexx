#pragma once
#include <skimagexx/array_views.hpp>
#include <skimagexx/slice_along_axes.hpp>
#include <skimagexx/image.hpp>
#include <cassert>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
inline void util_views_consumer() {
    using namespace skimagexx;using I=std::ptrdiff_t;
    const auto saved=get_num_threads();assert(saved==1);std::size_t checked=0;
    for(auto workers:{1,4}) {
        set_num_threads(workers);
        for(I rows=1;rows<=6;++rows)for(I cols=1;cols<=8;++cols)for(I word:{1,2,3,8,16}) {
            std::vector<unsigned char> input(std::size_t(rows*cols*word));
            for(std::size_t i=0;i<input.size();++i)input[i]=static_cast<unsigned char>(i*17+3);
            const auto unchanged=input;
            for(int layout=0;layout<4;++layout) {
                CropView a{input.data(),{rows,cols},{cols*word,word},std::size_t(word)};
                if(layout==1)a.strides={word,rows*word};
                if(layout==2){a.data=input.data()+(rows-1)*cols*word;a.strides[0]=-cols*word;}
                if(layout==3){a.data=input.data()+(cols-1)*word;a.strides[1]=-word;}
                for(I br=1;br<=rows;++br)if(rows%br==0)for(I bc=1;bc<=cols;++bc)if(cols%bc==0) {
                    auto b=view_as_blocks(a,{br,bc});assert(b.data==a.data&&b.itemsize==a.itemsize);
                    assert((b.shape==std::vector<I>{rows/br,cols/bc,br,bc}));
                    assert((b.strides==std::vector<I>{a.strides[0]*br,a.strides[1]*bc,a.strides[0],a.strides[1]}));
                    for(I i=0;i<rows/br;++i)for(I j=0;j<cols/bc;++j)for(I x=0;x<br;++x)for(I y=0;y<bc;++y) {
                        auto p=static_cast<const char*>(b.data)+i*b.strides[0]+j*b.strides[1]+x*b.strides[2]+y*b.strides[3];
                        auto expected=static_cast<const char*>(a.data)+(i*br+x)*a.strides[0]+(j*bc+y)*a.strides[1];assert(p==expected);assert(std::memcmp(p,expected,std::size_t(word))==0);++checked;
                    }
                }
                for(I wr=1;wr<=rows;++wr)for(I wc=1;wc<=cols;++wc)for(I step:{1,2,3}) {
                    auto b=view_as_windows(a,{wr,wc},{step,step});
                    assert((b.shape==std::vector<I>{(rows-wr)/step+1,(cols-wc)/step+1,wr,wc}));
                    for(I i=0;i<b.shape[0];++i)for(I j=0;j<b.shape[1];++j)for(I x=0;x<wr;++x)for(I y=0;y<wc;++y) {
                        auto p=static_cast<const char*>(b.data)+i*b.strides[0]+j*b.strides[1]+x*b.strides[2]+y*b.strides[3];
                        auto expected=static_cast<const char*>(a.data)+(i*step+x)*a.strides[0]+(j*step+y)*a.strides[1];assert(p==expected);assert(std::memcmp(p,expected,std::size_t(word))==0);++checked;
                    }
                }
                for(I first=0;first<=rows;++first)for(I last=first;last<=rows;++last) {
                    auto b=slice_along_axes(a,{{first,last}},nullptr,false);assert(b.kind==SliceAlongAxesResult::Kind::borrowed);assert((b.view().shape==std::vector<I>{last-first,cols}));
                    auto c=slice_along_axes(a,{{first,last}},nullptr,true);assert(c.kind==SliceAlongAxesResult::Kind::owned);
                    auto copied=c;auto moved=std::move(copied);assert(moved.view().data==(moved.payload.empty()?nullptr:moved.payload.data()));
                    for(I i=0;i<last-first;++i)for(I j=0;j<cols;++j) {
                        auto p=static_cast<const char*>(moved.view().data)+i*cols*word+j*word;
                        auto expected=static_cast<const char*>(a.data)+(first+i)*a.strides[0]+j*a.strides[1];assert(std::memcmp(p,expected,std::size_t(word))==0);++checked;
                    }
                }
                auto identity=slice_along_axes(a,{},nullptr,true);assert(identity.kind==SliceAlongAxesResult::Kind::identity&&identity.view().data==a.data);
                auto neg=slice_along_axes(a,{{-rows,rows}},nullptr,false);assert(neg.view().shape==a.shape);
                assert(input==unchanged&&get_num_threads()==workers);
            }
        }
    }
    CropView empty{nullptr,{0,4},{8,2},2};auto e=view_as_blocks(empty,{1,2});assert((e.shape==std::vector<I>{0,2,1,2}));
    auto copy=slice_along_axes(empty,{{0,0}},nullptr,true);assert(copy.payload.empty());
    CropView scalar{nullptr,{}, {}, 1};assert(view_as_blocks(scalar,{}).shape.empty());assert(view_as_windows(scalar,{},{}).shape.empty());
    auto throws=[&](auto f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}assert(caught);++checked;};
    throws([&]{view_as_blocks(empty,{0,1});});throws([&]{view_as_blocks(empty,{1});});
    throws([&]{view_as_windows(empty,{1,1},{1,1});});
    CropView a{nullptr,{4,4},{8,2},2};throws([&]{view_as_windows(a,{2,2},{0,1});});
    std::vector<I> duplicate{0,0},negative{-1};throws([&]{slice_along_axes(a,{{0,1},{0,1}},&duplicate);});throws([&]{slice_along_axes(a,{{0,1}},&negative);});
    throws([&]{slice_along_axes(a,{{3,1}});});throws([&]{slice_along_axes(empty,{{-1,0}});});
    CropView large{nullptr,{2},{std::numeric_limits<I>::max()},1};throws([&]{view_as_blocks(large,{2});});
    std::fenv_t environment;assert(std::fegetenv(&environment)==0);
    for(auto rounding:{FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}) {
        assert(std::fesetround(rounding)==0);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO|FE_INEXACT);
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);(void)view_as_blocks(a,{2,2});(void)view_as_windows(a,{2,2},{1,1});(void)slice_along_axes(a,{{0,2}});
        assert(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);++checked;
    }
    assert(std::fesetenv(&environment)==0);set_num_threads(saved);assert(get_num_threads()==saved);
    std::cout<<"PASS util_views complete_operations controls="<<checked<<" workers1_4_restored=1 serial_metadata=1\n";
}
