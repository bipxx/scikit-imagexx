#pragma once
#include <skimagexx/draw_multiblock_lbp.hpp>
#include <skimagexx/image.hpp>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
inline void draw_multiblock_lbp_consumer(){
    using namespace skimagexx;assert(get_num_threads()==1);
    std::vector<float> zeros(27,0);MultiblockLbpInput in{{zeros.data(),{3,3,3},{36,12,4},4},PixelType::float32,false};
    MultiblockLbpParameters p;p.r=-3;p.c=-3;p.width=p.height=2;p.code=170;p.alpha=.5;p.less={{0,0,0},{3}};
    for(int workers:{1,4}){set_num_threads(workers);auto a=draw_multiblock_lbp(in,p);assert(a.captured_workers==workers&&a.shape==std::vector<std::ptrdiff_t>({3,3,3})&&a.is_float32);for(int k=0;k<3;++k)assert(a.float32[12+k]==.9375f);assert(zeros==std::vector<float>(27,0));
        p.simd=false;auto b=draw_multiblock_lbp(in,p);assert(a.float32==b.float32);p.simd=true;}
    std::vector<std::uint8_t> white(12,255);MultiblockLbpInput gray{{white.data(),{3,4},{4,1},1},PixelType::uint8,false};p.r=p.c=0;p.height=p.width=0;auto copy=draw_multiblock_lbp(gray,p);assert(copy.shape==std::vector<std::ptrdiff_t>({3,4,3}));for(double x:copy.float64)assert(x==1.);
    // Signed-stride input and swapped integer transport: owning copy, not alias.
    std::uint16_t big[4]={0xff00,0,0xff00,0};MultiblockLbpInput reversed{{big+2,{2,2},{-4,2},2},PixelType::uint16,true};auto rev=draw_multiblock_lbp(reversed,p);assert(rev.float64[0]==255.*(1./65535.)&&rev.float64[3]==0.);
    p.width=1;p.height=1;p.greater={{0,0},{2}};p.less=p.greater;bool failed=false;try{(void)draw_multiblock_lbp(gray,p);}catch(const std::invalid_argument&){failed=true;}assert(failed);
    p=MultiblockLbpParameters{};p.r=PTRDIFF_MAX;p.height=1;failed=false;try{(void)draw_multiblock_lbp(gray,p);}catch(const std::overflow_error&){failed=true;}assert(failed);
    std::vector<std::uint8_t> large(512*513,255);p=MultiblockLbpParameters{};p.width=p.height=0;
    auto many=draw_multiblock_lbp({{large.data(),{512,513},{513,1},1},PixelType::uint8,false},p);assert(many.elements()==512*513*3);for(double x:many.float64)assert(x==1.);
    // Exercise the public compositional phase kernel itself, including its
    // unaligned SSE/scalar buffers and real per-operation exception status.
    std::vector<unsigned char> input(1+17*4),output(1+17*4);float factor=.25f;
    for(int k=0;k<17;++k){float value=float(k+1);std::memcpy(input.data()+1+4*k,&value,4);}
    MultiblockLbpFloatView x{{input.data()+1,{17},{4},4},false},y{{&factor,{},{},4},false},z{{output.data()+1,{17},{4},4},false};
    assert(multiblock_lbp_binary(x,y,z,false)==0);
    for(int k=0;k<17;++k){float value;std::memcpy(&value,output.data()+1+4*k,4);assert(value==float(k+1)*.25f);}
    std::vector<float> reversed_phase(17);MultiblockLbpFloatView reversed_output{{reversed_phase.data()+16,{17},{-4},4},false};multiblock_lbp_assign(reversed_output,z);
    for(int k=0;k<17;++k)assert(reversed_phase[16-k]==float(k+1)*.25f);
    double inf=std::numeric_limits<double>::infinity(),zero=0,invalid_result=0;
    assert(multiblock_lbp_binary({{&inf,{},{},8},false},{{&zero,{},{},8},false},{{&invalid_result,{},{},8},false},false)==8);assert(std::isnan(invalid_result));
    std::uint16_t half_input[2]={0x3c00,0x4000},half_factor=0x3800,half_output[2]={0,0};
    assert(multiblock_lbp_binary({{half_input,{2},{2},2},false},{{&half_factor,{},{},2},false},{{half_output,{2},{2},2},false},false)==0);assert(half_output[0]==0x3800&&half_output[1]==0x3c00);
    set_num_threads(1);std::cout<<"PASS owning-copy/gray/normalization/ordered-physical-overlap/scalar-SSE2/budgets/byteswap/stride/broadcast/overflow invariants; NOT original parity\n";
}
