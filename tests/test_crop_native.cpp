#include <skimagexx/crop.hpp>
#include <skimagexx/image.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
int main() {
    using namespace skimagexx;std::int16_t a[20];for(int i=0;i<20;++i)a[i]=std::int16_t(i);
    CropView input{a,{4,5},{10,2},2};auto plan=crop_view(input,{{1,3},{1,4}});
    assert((plan.view.shape==std::vector<std::ptrdiff_t>{2,3}));assert(plan.view.data==a+6);
    std::int16_t out[6]={};crop_copy_into(input,{{1,3},{1,4}},out,{6,2});
    for(int i=0;i<6;++i)assert(out[i]==a[6+(i/3)*5+i%3]);
    std::int16_t column[6]={};crop_copy_into(input,{{1,3},{1,4}},column,{2,4});
    for(int r=0;r<2;++r)for(int c=0;c<3;++c)assert(column[r+c*2]==a[(r+1)*5+c+1]);
    CropView reverse{a+19,{4,5},{-10,-2},2};crop_copy_into(reverse,{{1,3},{1,4}},out,{6,2});
    for(int i=0;i<6;++i)assert(out[i]==a[13-(i/3)*5-i%3]);
    auto negative=crop_view(input,{{-3,-1},{-4,-1}});assert(negative.view.shape==plan.view.shape&&negative.view.data==plan.view.data);
    auto empty=crop_view(input,{{9,-9},{0,5}});assert(empty.view.shape[0]==0);crop_copy_into(input,{{9,-9},{0,5}},nullptr,{10,2});
    std::uint64_t word=0xfff8000000000042ULL,copy=0;CropView scalar{&word,{}, {},8};crop_copy_into(scalar,{},&copy,{});assert(copy==word);
    // Exercise byte widths including complex128 without arithmetic/coercion.
    for(std::size_t size: {1u,2u,3u,4u,6u,8u,16u}) {
        std::vector<unsigned char> bytes(12*size),dest(6*size);
        for(std::size_t i=0;i<bytes.size();++i)bytes[i]=static_cast<unsigned char>(i*17);
        CropView v{bytes.data(),{3,4},{std::ptrdiff_t(4*size),std::ptrdiff_t(size)},size};
        crop_copy_into(v,{{0,2},{1,4}},dest.data(),{std::ptrdiff_t(3*size),std::ptrdiff_t(size)});
        for(std::size_t r=0;r<2;++r)for(std::size_t i=0;i<3*size;++i)assert(dest[r*3*size+i]==bytes[r*4*size+size+i]);
    }
    assert(get_num_threads()==1);set_num_threads(4);crop_copy_into(input,{{1,3},{1,4}},out,{6,2});set_num_threads(1);
    std::cout<<"PASS crop installed SDK view+copy; serial/default1; SIMD-independent "<<has_avx2()<<"\n";
}
