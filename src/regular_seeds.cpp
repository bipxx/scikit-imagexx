#include "regular_seeds.hpp"
#include "execution.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace skimagexx::detail {
namespace {
template<class T> void assign(const RegularSeedsGrid& grid, T* output, int workers) {
    const bool vectorized=has_avx2();
    partition(grid.selected_count,grid.selected_count>=parallel_min?workers:1,
              [&](std::size_t begin,std::size_t end,auto) {
        while(begin<end) {
            std::size_t remaining=begin,offset=0,run=end-begin,stride=1;
            for(std::size_t axis=grid.ndim;axis-->0;) {
                const auto& a=grid.axes[axis];
                const auto coordinate=remaining%a.count;
                remaining/=a.count;
                offset+=(a.start+coordinate*a.step)*a.output_stride;
                if(axis+1==grid.ndim) {
                    run=std::min(run,a.count-coordinate);
                    stride=a.step*a.output_stride;
                }
            }
            T* destination=output+offset;
#ifdef SKIMAGEXX_AVX2
            if constexpr(std::is_same_v<T,std::uint32_t>) {
                if(vectorized && stride==1 && run>=8) {
                    simd::regular_seeds_u32_run(destination,run,begin+1);
                    begin+=run;continue;
                }
            }
            if constexpr(std::is_same_v<T,std::uint64_t>) {
                if(vectorized && stride==1 && run>=4) {
                    simd::regular_seeds_u64_run(destination,run,begin+1);
                    begin+=run;continue;
                }
            }
#else
            (void)vectorized;
#endif
            for(std::size_t i=0;i<run;++i)
                destination[i*stride]=static_cast<T>(begin+i+1);
            begin+=run;
        }
    });
}
}
void regular_seeds_assign(const RegularSeedsGrid& grid,void* output,
                          RegularSeedsType type,int workers) {
    if(!grid.selected_count)return;
    if(type==RegularSeedsType::boolean) {
        partition(grid.selected_count,grid.selected_count>=parallel_min?workers:1,
                  [&](std::size_t begin,std::size_t end,auto) {
            for(auto index=begin;index<end;++index) {
                auto remaining=index;std::size_t offset=0;
                for(std::size_t axis=grid.ndim;axis-->0;) {
                    const auto& a=grid.axes[axis];
                    offset+=(a.start+(remaining%a.count)*a.step)*a.output_stride;
                    remaining/=a.count;
                }
                static_cast<std::uint8_t*>(output)[offset]=1;
            }
        });
        return;
    }
    switch(type) {
    case RegularSeedsType::u8:case RegularSeedsType::i8:assign(grid,static_cast<std::uint8_t*>(output),workers);break;
    case RegularSeedsType::u16:case RegularSeedsType::i16:assign(grid,static_cast<std::uint16_t*>(output),workers);break;
    case RegularSeedsType::u32:case RegularSeedsType::i32:assign(grid,static_cast<std::uint32_t*>(output),workers);break;
    case RegularSeedsType::u64:case RegularSeedsType::i64:assign(grid,static_cast<std::uint64_t*>(output),workers);break;
    case RegularSeedsType::f32:assign(grid,static_cast<float*>(output),workers);break;
    case RegularSeedsType::f64:assign(grid,static_cast<double*>(output),workers);break;
    case RegularSeedsType::boolean:break;
    }
}
void regular_seeds_into(const RegularSeedsGrid& grid,void* output,
                        RegularSeedsType type,int workers) {
    std::size_t width=1;
    switch(type) {
    case RegularSeedsType::u16:case RegularSeedsType::i16:width=2;break;
    case RegularSeedsType::u32:case RegularSeedsType::i32:case RegularSeedsType::f32:width=4;break;
    case RegularSeedsType::u64:case RegularSeedsType::i64:case RegularSeedsType::f64:width=8;break;
    default:break;
    }
    const auto bytes=grid.output_count*width;
    const bool vectorized=has_avx2();
    partition(bytes,bytes>=parallel_min?workers:1,[&](auto begin,auto end,auto) {
        if(begin==end)return;
#ifdef SKIMAGEXX_AVX2
        if(vectorized && end-begin>=32)simd::regular_seeds_zero(static_cast<char*>(output)+begin,end-begin);
        else
#else
        (void)vectorized;
#endif
            std::memset(static_cast<char*>(output)+begin,0,end-begin);
    });
    regular_seeds_assign(grid,output,type,workers);
}
}
