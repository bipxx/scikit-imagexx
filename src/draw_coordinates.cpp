#include "skimagexx/draw_coordinates.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cstring>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
#pragma STDC FENV_ACCESS ON

#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t draw_fill(std::intptr_t*,std::size_t,std::intptr_t,bool);
std::size_t draw_ellipse_row(const double*,const double*,std::size_t,double,double,
                             double,double,std::uint8_t*);
}
#endif
namespace skimagexx::detail {
namespace {
void coordinate(DrawCoordinatePair output,std::size_t i,std::intptr_t r,std::intptr_t c) {
    output.rows[std::ptrdiff_t(i)*output.row_step]=r;
    output.columns[std::ptrdiff_t(i)*output.column_step]=c;
}
bool finite(double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
    return (bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
}
class Capture {
    std::fenv_t saved_;std::atomic<bool>& refused_;bool active_,ready_;
public:
    explicit Capture(std::atomic<bool>& refused):refused_(refused),active_(std::feholdexcept(&saved_)==0),ready_(active_&&std::fegetround()==FE_TONEAREST) {
#if defined(__SSE2__) || defined(_M_X64)
        if(_mm_getcsr()&0x8040)ready_=false;
#endif
        if(!ready_)refused_.store(true,std::memory_order_relaxed);
    }
    bool ready()const{return ready_;}
    ~Capture(){if(active_){if(std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW))refused_.store(true,std::memory_order_relaxed);std::fesetenv(&saved_);}}
};
void fill(std::intptr_t* output,std::size_t count,std::intptr_t first,bool ramp,bool vectorized) {
    std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
    if(vectorized)done=simd::draw_fill(output,count,first,ramp);
#else
    (void)vectorized;
#endif
    for(;done<count;++done)output[done]=ramp?first+std::intptr_t(done):first;
}
}

void draw_line_into(std::intptr_t r0,std::intptr_t c0,std::intptr_t r1,
                    std::intptr_t c1,DrawCoordinatePair output,int workers) {
    (void)workers; // Exact dependent recurrence; do not race octant/tie state.
    auto r=r0,c=c0;
    auto dr=r1>=r0?r1-r0:r0-r1,dc=c1>=c0?c1-c0:c0-c1;
    auto sr=r1>r0?std::intptr_t(1):std::intptr_t(-1);
    auto sc=c1>c0?std::intptr_t(1):std::intptr_t(-1);
    const bool steep=dr>dc;
    if(steep){std::swap(c,r);std::swap(dc,dr);std::swap(sc,sr);}
    auto error=2*dr-dc;
    for(std::intptr_t i=0;i<dc;++i) {
        coordinate(output,std::size_t(i),steep?c:r,steep?r:c);
        while(error>=0){r+=sr;error-=2*dc;}
        c+=sc;error+=2*dr;
    }
    coordinate(output,std::size_t(dc),r1,c1);
}

void draw_rectangle_into(const std::intptr_t* starts,const std::size_t* lengths,
                         std::size_t ndim,std::size_t count,
                         std::intptr_t* const* outputs,int workers) {
    if(!count)return;
    std::vector<std::size_t> shape(lengths,lengths+ndim),pitch(ndim,1);
    if(ndim>1)std::swap(shape[0],shape[1]);
    for(auto d=ndim;d-->1;)pitch[d-1]=pitch[d]*shape[d];
    const bool vectorized=has_avx2();
    for(std::size_t d=0;d<ndim;++d) {
        const auto axis=ndim>1&&d<2?1-d:d,step=pitch[axis],length=lengths[d];
        partition(count,count>=parallel_min?workers:1,[&](auto begin,auto end,auto) {
            auto i=begin;
            while(i<end) {
                const auto index=(i/step)%length;
                const auto run=std::min(end-i,step==1?length-index:step-i%step);
                fill(outputs[d]+i,run,starts[d]+std::intptr_t(index),step==1,vectorized);
                i+=run;
            }
        });
    }
}

bool draw_ellipse_mask_into(const DrawEllipseGrid& grid,std::uint8_t* mask,
                            std::size_t* row_prefix,int workers) {
    if(!finite(grid.center_row)||!finite(grid.center_column)||!finite(grid.row_radius)||
       !finite(grid.column_radius)||!finite(grid.sin_alpha)||!finite(grid.cos_alpha))return false;
    if(grid.row_radius<=0||grid.column_radius<=0)return false;
    std::atomic<bool> refused{false};
    {
        Capture capture(refused);if(!capture.ready())return false;
        // These are exactly the source's two broadcast column products. Also
        // evaluate them for zero rows: NumPy evaluates operands before broadcast.
        std::vector<double> column_sin(grid.columns),column_cos(grid.columns);
        for(std::size_t c=0;c<grid.columns;++c) {
            const double delta=double(c)-grid.center_column;
            column_sin[c]=delta*grid.sin_alpha;column_cos[c]=delta*grid.cos_alpha;
        }
        const bool vectorized=has_avx2();
        const auto count=grid.rows*grid.columns;
        partition(grid.rows,count>=parallel_min?workers:1,[&](auto begin,auto end,auto) {
            Capture local(refused);if(!local.ready())return;
            for(auto r=begin;r<end;++r) {
                const double delta=double(r)-grid.center_row;
                const double row_cos=delta*grid.cos_alpha,row_sin=delta*grid.sin_alpha;
                std::size_t hits=0,done=0;
#ifdef SKIMAGEXX_AVX2
                if(vectorized&&grid.columns>=4) {
                    done=grid.columns/4*4;
                    hits=simd::draw_ellipse_row(column_sin.data(),column_cos.data(),done,
                        row_cos,row_sin,grid.row_radius,grid.column_radius,mask+r*grid.columns);
                }
#else
                (void)vectorized;
#endif
                for(auto c=done;c<grid.columns;++c) {
                    const double a=(row_cos+column_sin[c])/grid.row_radius;
                    const double b=(row_sin-column_cos[c])/grid.column_radius;
                    const auto inside=std::uint8_t(a*a+b*b<1.0);
                    mask[r*grid.columns+c]=inside;hits+=inside;
                }
                row_prefix[r+1]=hits;
            }
        });
    }
    if(refused.load(std::memory_order_relaxed))return false;
    row_prefix[0]=0;
    for(std::size_t r=0;r<grid.rows;++r)row_prefix[r+1]+=row_prefix[r];
    return true;
}

void draw_ellipse_emit_into(const DrawEllipseGrid& grid,const std::uint8_t* mask,
                            const std::size_t* row_prefix,
                            DrawCoordinatePair output,int workers) {
    const auto count=grid.rows*grid.columns;
    partition(grid.rows,count>=parallel_min?workers:1,[&](auto begin,auto end,auto) {
        for(auto r=begin;r<end;++r) {
            auto out=row_prefix[r];
            for(std::size_t c=0;c<grid.columns;++c)if(mask[r*grid.columns+c])
                coordinate(output,out++,grid.row_offset+std::intptr_t(r),grid.column_offset+std::intptr_t(c));
        }
    });
}
}
