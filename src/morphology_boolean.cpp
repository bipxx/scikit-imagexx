#include "morphology_detail.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <memory>

namespace skimagexx::detail {
namespace {
void insert(std::uint64_t* output,std::size_t index,std::uint64_t bits,std::size_t count) {
    const auto word=index/64,shift=index%64;
    output[word]|=bits<<shift;
    if (shift+count>64) output[word+1]|=bits>>(64-shift);
}
bool pack(const std::uint8_t* source,std::uint64_t* output,std::size_t n,std::size_t offset) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) return simd::boolean_pack(source,output,n,offset);
#endif
    for (std::size_t j=0;j<n;++j) {
        if (source[j]>1) return false;
        if (source[j]) insert(output,j+offset,1,1);
    }
    return true;
}
void combine(const std::uint64_t* a,const std::uint64_t* b,std::uint64_t* out,std::size_t n,bool erode) {
    std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) i=simd::boolean_combine(a,b,out,n,erode);
#endif
    for (;i<n;++i) out[i]=erode ? a[i]&b[i] : a[i]|b[i];
}
void shift_reduce(std::uint64_t* data,std::size_t n,std::size_t shift,bool erode) {
    std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) i=simd::boolean_shift_reduce(data,n,shift,erode);
#endif
    const auto words=shift/64,bits=shift%64;
    for (;i<n;++i) {
        std::uint64_t value=0;
        if (i+words<n) {
            value=data[i+words]>>bits;
            if (bits && i+words+1<n) value|=data[i+words+1]<<(64-bits);
        }
        data[i]=erode ? data[i]&value : data[i]|value;
    }
}
void unpack(const std::uint64_t* input,std::uint8_t* output,std::size_t n) {
    std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) i=simd::boolean_unpack(input,output,n);
#endif
    for (;i<n;++i) output[i]=std::uint8_t((input[i/64]>>(i%64))&1);
}
}
// The input must contain only raw 0/1 bytes. Reject noncanonical bool storage
// before writing output so the general grayscale path can preserve its bytes.
bool boolean_rectangle_into(const std::uint8_t* source,std::uint8_t* output,std::size_t height,std::size_t width,
    const std::vector<std::ptrdiff_t>& rows,const std::vector<std::ptrdiff_t>& cols,bool fill,bool erode,int workers) {
    const auto extended=rows.size(),words=(cols.size()+63)/64;
    if (extended>std::numeric_limits<std::size_t>::max()/words/sizeof(std::uint64_t)) return false;
    const auto count=extended*words,vertical=extended-height+1,horizontal=cols.size()-width+1;
    const auto offset=(cols.size()-width)/2;
    std::unique_ptr<std::uint64_t[]> image(new std::uint64_t[count]),prefix(new std::uint64_t[count]),suffix(new std::uint64_t[count]);
    std::vector<unsigned char> invalid(std::size_t(workers),0);
    partition(extended,workers,[&](auto begin,auto end,auto id) {
        bool valid=true;
        for (auto y=begin;y<end;++y) {
            auto* line=image.get()+y*words;
            if (rows[y]<0) { std::fill(line,line+words,fill ? ~std::uint64_t(0) : 0); continue; }
            std::fill(line,line+words,0);
            const auto* input=source+std::size_t(rows[y])*width;
            if (!pack(input,line,width,offset)) { valid=false; continue; }
            for (std::size_t j=0;j<offset;++j) {
                const auto left=cols[j]<0 ? std::uint8_t(fill) : input[std::size_t(cols[j])];
                const auto right_index=j+width+offset;
                const auto right=cols[right_index]<0 ? std::uint8_t(fill) : input[std::size_t(cols[right_index])];
                if (left) insert(line,j,1,1);
                if (right) insert(line,right_index,1,1);
            }
        }
        invalid[id]=static_cast<unsigned char>(!valid);
    });
    if (std::any_of(invalid.begin(),invalid.end(),[](auto v){return v!=0;})) return false;
    for (std::size_t block=0;block<extended;block+=vertical) {
        const auto stop=std::min(extended,block+vertical);
        std::memcpy(prefix.get()+block*words,image.get()+block*words,words*sizeof(std::uint64_t));
        for (auto y=block+1;y<stop;++y) combine(prefix.get()+(y-1)*words,image.get()+y*words,prefix.get()+y*words,words,erode);
        std::memcpy(suffix.get()+(stop-1)*words,image.get()+(stop-1)*words,words*sizeof(std::uint64_t));
        for (auto y=stop-1;y>block;--y) combine(suffix.get()+y*words,image.get()+(y-1)*words,suffix.get()+(y-1)*words,words,erode);
    }
    partition(height,workers,[&](auto begin,auto end,auto) {
        std::unique_ptr<std::uint64_t[]> line(new std::uint64_t[words]);
        for (auto y=begin;y<end;++y) {
            combine(suffix.get()+y*words,prefix.get()+(y+vertical-1)*words,line.get(),words,erode);
            std::size_t span=1;
            while (span<=horizontal/2) { shift_reduce(line.get(),words,span,erode); span*=2; }
            if (span<horizontal) shift_reduce(line.get(),words,horizontal-span,erode);
            unpack(line.get(),output+y*width,width);
        }
    });
    return true;
}
}
