#include "skimagexx/draw_polygon.hpp"
#include "skimagexx/image.hpp"
#include "polygon.hpp"
#include "execution.hpp"
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace skimagexx::detail {
namespace {
constexpr std::intptr_t coordinate_limit = 1<<20;
bool valid_workers(int workers) noexcept {
    if(workers<1 || workers>1024)return false;
#ifndef _OPENMP
    if(workers!=1)return false;
#endif
    return true;
}
bool coordinate(double value) noexcept {
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
    bits&=UINT64_C(0x7fffffffffffffff);
    return bits==0 || (bits>=UINT64_C(0x37f0000000000000) && bits<=UINT64_C(0x4130000000000000));
}
class Environment {
    std::fenv_t saved_;
    bool valid_;
public:
    Environment():valid_(std::feholdexcept(&saved_)==0) {
        if(valid_)std::fesetround(FE_TONEAREST);
    }
    ~Environment(){if(valid_)std::fesetenv(&saved_);}
    bool valid()const noexcept{return valid_;}
};
}

bool draw_polygon_prepare(const DrawPolygonVertices& vertices,const DrawPolygonScan& scan,
                          DrawPolygonPlan& output,int workers) {
    if(!valid_workers(workers) || !vertices.count || vertices.count>draw_polygon_max_vertices
       || !vertices.rows || !vertices.columns || std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    if(_mm_getcsr()&((1u<<15)|(1u<<6)|(3u<<13)))return false;
#endif
    for(std::size_t i=0;i<vertices.count;++i)
        if(!coordinate(vertices.rows[i]) || !coordinate(vertices.columns[i]))return false;
    DrawPolygonPlan plan;plan.first=scan.first;
    for(std::size_t d=0;d<2;++d)
        if(scan.first[d]<0 || scan.first[d]>coordinate_limit || scan.last[d]>coordinate_limit)return false;
    if(scan.last[0]<scan.first[0] || scan.last[1]<scan.first[1]) {
        plan.row_offsets.push_back(0);output=std::move(plan);return true;
    }
    for(std::size_t d=0;d<2;++d)plan.shape[d]=std::size_t(scan.last[d]-scan.first[d])+1;
    if(plan.shape[0]>draw_polygon_max_scan/plan.shape[1])return false;
    const auto count=plan.shape[0]*plan.shape[1];
    if(count>draw_polygon_max_edge_tests/vertices.count)return false;
    plan.labels.resize(count);plan.row_offsets.resize(plan.shape[0]+1,0);
    // Geometry.pyx is column-first: point_in_polygon(cptr,rptr,c_i,r_i).
    const PolygonView polygon{vertices.columns,vertices.rows,vertices.count};
    const bool vectorized=has_avx2();std::atomic<bool> failed{false};
    partition(count,count>=parallel_min?workers:1,[&](std::size_t begin,std::size_t end,std::size_t) {
        Environment environment;
        if(!environment.valid()){failed.store(true,std::memory_order_relaxed);return;}
        std::array<double,128> x{},y{};
        while(begin<end) {
            const auto row=begin/plan.shape[1],column=begin%plan.shape[1];
            const auto length=std::min({end-begin,plan.shape[1]-column,x.size()});
            const double r=double(plan.first[0]+std::intptr_t(row));
#ifdef SKIMAGEXX_AVX2
            if(vectorized) {
                for(std::size_t i=0;i<length;++i){x[i]=double(plan.first[1]+std::intptr_t(column+i));y[i]=r;}
                simd::polygon_points(polygon,x.data(),y.data(),plan.labels.data()+begin,0,length);
            } else
#else
            (void)vectorized;(void)y;
#endif
            {
                for(std::size_t i=0;i<length;++i)
                    plan.labels[begin+i]=polygon_label(polygon,double(plan.first[1]+std::intptr_t(column+i)),r);
            }
            begin+=length;
        }
    });
    if(failed.load(std::memory_order_relaxed))throw std::runtime_error("cannot capture polygon floating environment");
    partition(plan.shape[0],count>=parallel_min?workers:1,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto r=begin;r<end;++r) {
            std::size_t found=0;
            for(std::size_t c=0;c<plan.shape[1];++c)found+=plan.labels[r*plan.shape[1]+c]!=0;
            plan.row_offsets[r+1]=found;
        }
    });
    for(std::size_t r=1;r<plan.row_offsets.size();++r)plan.row_offsets[r]+=plan.row_offsets[r-1];
    plan.count=plan.row_offsets.back();output=std::move(plan);return true;
}

void draw_polygon_coords_into(const DrawPolygonPlan& plan,std::intptr_t* rows,
                             std::intptr_t* columns,int workers) {
    if(!valid_workers(workers) || (plan.count && (!rows || !columns)))
        throw std::invalid_argument("polygon coordinate output/worker boundary");
    partition(plan.shape[0],plan.labels.size()>=parallel_min?workers:1,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto r=begin;r<end;++r) {
            auto target=plan.row_offsets[r];
            for(std::size_t c=0;c<plan.shape[1];++c)if(plan.labels[r*plan.shape[1]+c]!=0) {
                rows[target]=plan.first[0]+std::intptr_t(r);
                columns[target]=plan.first[1]+std::intptr_t(c);++target;
            }
        }
    });
}

void draw_polygon_mask_into(const DrawPolygonPlan& plan,std::size_t rows,std::size_t columns,
                           std::uint8_t* output,int workers) {
    const auto limit=std::size_t(std::numeric_limits<std::intptr_t>::max());
    if(!valid_workers(workers) || rows>limit || columns>limit
       || (columns && rows>draw_polygon_max_scan/columns))
        throw std::invalid_argument("polygon mask shape/worker boundary");
    const auto count=rows*columns;
    if(count && !output)throw std::invalid_argument("polygon mask requires output");
    // New mask geometry is an SDK boundary. Only inspect selected points if
    // the scan rectangle extends outside it; never clear before rejecting.
    if(plan.count && (std::size_t(plan.first[0])+plan.shape[0]>rows
                    || std::size_t(plan.first[1])+plan.shape[1]>columns)) {
        for(std::size_t r=0;r<plan.shape[0];++r)for(std::size_t c=0;c<plan.shape[1];++c)
            if(plan.labels[r*plan.shape[1]+c]!=0
               && (std::size_t(plan.first[0])+r>=rows || std::size_t(plan.first[1])+c>=columns))
                throw std::invalid_argument("polygon selected coordinate outside mask");
    }
    partition(count,count>=parallel_min?workers:1,[&](std::size_t begin,std::size_t end,std::size_t) {
        if(begin<end)std::fill(output+begin,output+end,std::uint8_t(0));
    });
    partition(plan.shape[0],plan.labels.size()>=parallel_min?workers:1,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto r=begin;r<end;++r)for(std::size_t c=0;c<plan.shape[1];++c)
            if(plan.labels[r*plan.shape[1]+c]!=0)
                output[(std::size_t(plan.first[0])+r)*columns+std::size_t(plan.first[1])+c]=1;
    });
}
}
