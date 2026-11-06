// scikit-image0.26.0 _draw.pyx: rational quadratic recurrence, BSD-3-Clause.
// See third_party/scikit-image-LICENSE.txt. Source association/order is literal.
#include <skimagexx/bezier_curve.hpp>
#include "execution.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <utility>
#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
#ifdef SKIMAGEXX_BEZIER_TRACE
#include <cstdio>
#define BEZIER_TRACE(...) std::fprintf(stderr,__VA_ARGS__)
#else
#define BEZIER_TRACE(...) ((void)0)
#endif

namespace skimagexx::detail {
namespace {
using Int=std::intptr_t;
constexpr Int coordinate_limit=Int(1)<<21;
constexpr std::size_t point_limit=1000000;
struct Environment {
    std::fenv_t saved;
    Environment() {std::feholdexcept(&saved);}
    ~Environment() {std::fesetenv(&saved);}
};
// Cython int casts truncate, including floor(+.5) results. Bound before cast.
bool cast_coordinate(double value,Int& output) {
    if(!std::isfinite(value)||value < -double(coordinate_limit)||value > double(coordinate_limit))return false;
    output=static_cast<Int>(value);return true;
}
struct Builder {
    BezierCurve& p;
    std::size_t visited=0;
    bool emit(Int row,Int column) {
        if(++visited>point_limit || row < -coordinate_limit || row > coordinate_limit || column < -coordinate_limit || column > coordinate_limit)return false;
        const auto& q=p.request;
        if(!q.clipped || (row>=0 && column>=0 && double(row)<q.height && double(column)<q.width)) {
            p.rows.push_back(row);p.columns.push_back(column);
        }
        return true;
    }
    bool line(Int r0,Int c0,Int r2,Int c2) {
        const auto count=std::size_t(std::max(std::abs(r2-r0),std::abs(c2-c0)))+1;
        if(count>point_limit-visited)return false;
        // Reuse existing source-faithful line, including strict steep/tie order.
        const auto begin=p.rows.size();p.rows.resize(begin+count);p.columns.resize(begin+count);
        draw_line_into(r0,c0,r2,c2,{p.rows.data()+begin,p.columns.data()+begin,1,1},1);
        visited+=count;
        if(p.request.clipped) {
            auto write=begin;
            for(auto i=begin;i<begin+count;++i) {
                const auto r=p.rows[i],c=p.columns[i];
                if(r>=0 && c>=0 && double(r)<p.request.height && double(c)<p.request.width) {
                    p.rows[write]=r;p.columns[write]=c;++write;
                }
            }
            p.rows.resize(write);p.columns.resize(write);
        }
        return true;
    }
    bool segment(Int r0,Int c0,Int r1,Int c1,Int r2,Int c2,double weight) {
        BEZIER_TRACE("segment %lld %lld %lld %lld %lld %lld %.17g\n",(long long)r0,(long long)c0,(long long)r1,(long long)c1,(long long)r2,(long long)c2,weight);
        double sc=double(c2-c1),sr=double(r2-r1);
        double d2c=double(c0-c2),d2r=double(r0-r2),d1c=double(c0-c1),d1r=double(r0-r1);
        double rc=d1c*sr+d1r*sc,cur=d1c*sr-d1r*sc;
        if(cur!=0 && weight>0) {
            if(sc*sc+sr*sr>d1c*d1c+d1r*d1r) {
                c2=c0;c0-=static_cast<Int>(d2c);r2=r0;r0-=static_cast<Int>(d2r);cur=-cur;
            }
            d1c=2*(4*weight*sc*d1c+d2c*d2c);
            d1r=2*(4*weight*sr*d1r+d2r*d2r);
            sc=c0<c2?1:-1;sr=r0<r2?1:-1;
            rc=-2*sc*sr*(2*weight*rc+d2c*d2r);
            if(cur*sc*sr<0) {d1c=-d1c;d1r=-d1r;rc=-rc;cur=-cur;}
            d2c=4*weight*double(c1-c0)*sr*cur+d1c/2+rc;
            d2r=4*weight*double(r0-r1)*sc*cur+d1r/2+rc;
            if(weight<.5 && (d2r>rc || d2c<rc)) {
                cur=(weight+1)/2;weight=std::sqrt(weight);rc=1./(weight+1);
                sc=std::floor((double(c0)+2*weight*double(c1)+double(c2))*rc*.5+.5);
                sr=std::floor((double(r0)+2*weight*double(r1)+double(r2))*rc*.5+.5);
                d2c=std::floor((weight*double(c1)+double(c0))*rc+.5);
                d2r=std::floor((double(r1)*weight+double(r0))*rc+.5);
                Int mr,mc,cr,cc;
                if(!cast_coordinate(sr,mr)||!cast_coordinate(sc,mc)||!cast_coordinate(d2r,cr)||!cast_coordinate(d2c,cc))return false;
                BEZIER_TRACE("flat-half %.17g\n",cur);
                return segment(r0,c0,cr,cc,mr,mc,cur); // Literal upstream first-half return.
            }
            double err=d2c+d2r-rc;
            while(d2r<=rc && d2c>=rc) {
                if(!emit(r0,c0))return false;
                if(c0==c2 && r0==r2)return true;
                const bool test1=2*err>d2r,test2=2*(err+d1r)<-d2r;
                const auto old_r=r0,old_c=c0;
                if(2*err<d2c || test2) {r0+=static_cast<Int>(sr);d2r+=rc;d2c+=d1c;err+=d2c;}
                if(2*err>d2c || test1) {c0+=static_cast<Int>(sc);d2c+=rc;d2r+=d1r;err+=d2r;}
                if(r0==old_r && c0==old_c)return false;
            }
        }
        return line(r0,c0,r2,c2);
    }
    bool curve() {
        const auto& q0=p.request;
        Int r0=q0.r0,c0=q0.c0,r1=q0.r1,c1=q0.c1,r2=q0.r2,c2=q0.c2;
        double weight=q0.weight;
        // Source declares vc/vr C int, not Py_ssize_t. Bounds prove these sums
        // fit int32; later assignments go through the same truncating cast.
        Int vc=c0-2*c1+c2,vr=r0-2*r1+r2;
        double dc=double(c0-c1),dr=double(r0-r1),ww,t,q;
        if(dc*double(c2-c1)>0) {
            if(dr*double(r2-r1)) {
                if(std::abs(dc*double(vr))>std::abs(dr*double(vc))) {
                    c0=c2;c2=static_cast<Int>(dc+double(c1));r0=r2;r2=static_cast<Int>(dr+double(r1));
                }
            }
            if(c0==c2 || weight==1.)t=double(c0-c1)/double(vc);
            else {
                q=std::sqrt(4.*weight*weight*double(c0-c1)*double(c2-c1)+double(c2-c0)*std::floor(double(c2-c0)));
                if(c1<c0)q=-q;
                t=(2.*weight*double(c0-c1)-double(c0)+double(c2)+q)/(2.*(1.-weight)*double(c2-c0));
            }
            q=1./(2.*t*(1.-t)*(weight-1.)+1.0);
            dc=(t*t*(double(c0)-2.*weight*double(c1)+double(c2))+2.*t*(weight*double(c1)-double(c0))+double(c0))*q;
            dr=(t*t*(double(r0)-2.*weight*double(r1)+double(r2))+2.*t*(weight*double(r1)-double(r0))+double(r0))*q;
            ww=t*(weight-1.)+1.;ww*=ww*q;weight=((1.-t)*(weight-1.)+1.)*std::sqrt(q);
            if(!cast_coordinate(dc+.5,vc)||!cast_coordinate(dr+.5,vr))return false;
            dr=(dc-double(c0))*double(r1-r0)/double(c1-c0)+double(r0);
            Int control;
            if(!cast_coordinate(dr+.5,control))return false;
            BEZIER_TRACE("column-extremum t=%.17g q=%.17g\n",t,q);
            if(!segment(r0,c0,control,vc,vr,vc,ww))return false;
            dr=(dc-double(c2))*double(r1-r2)/double(c1-c2)+double(r2);
            if(!cast_coordinate(dr+.5,r1))return false;
            c0=c1=vc;r0=vr;
        }
        if(double(r0-r1)*std::floor(double(r2-r1))>0) {
            if(r0==r2 || weight==1)t=double(r0-r1)/(double(r0)-2.*double(r1)+double(r2));
            else {
                q=std::sqrt(4.*weight*weight*double(r0-r1)*double(r2-r1)+double(r2-r0)*std::floor(double(r2-r0)));
                if(r1<r0)q=-q;
                t=(2.*weight*double(r0-r1)-double(r0)+double(r2)+q)/(2.*(1.-weight)*double(r2-r0));
            }
            q=1./(2.*t*(1.-t)*(weight-1.)+1.);
            dc=(t*t*(double(c0)-2.*weight*double(c1)+double(c2))+2.*t*(weight*double(c1)-double(c0))+double(c0))*q;
            dr=(t*t*(double(r0)-2.*weight*double(r1)+double(r2))+2.*t*(weight*double(r1)-double(r0))+double(r0))*q;
            ww=t*(weight-1.)+1.;ww*=ww*q;weight=((1.-t)*(weight-1.)+1.)*std::sqrt(q);
            if(!cast_coordinate(dc+.5,vc)||!cast_coordinate(dr+.5,vr))return false;
            dc=double(c1-c0)*(dr-double(r0))/double(r1-r0)+double(c0);
            Int control;
            if(!cast_coordinate(dc+.5,control))return false;
            BEZIER_TRACE("row-extremum t=%.17g q=%.17g\n",t,q);
            if(!segment(r0,c0,vr,control,vr,vc,ww))return false;
            dc=double(c1-c2)*(dr-double(r2))/double(r1-r2)+double(c2);
            if(!cast_coordinate(dc+.5,c1))return false;
            c0=vc;r0=r1=vr;
        }
        return segment(r0,c0,r1,c1,r2,c2,weight*weight);
    }
};
}

bool prepare_bezier_curve(const BezierCurveRequest& q,BezierCurve& output) {
    if(sizeof(Int)!=8 || std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64)
    if(_mm_getcsr()&((3u<<13)|(1u<<15)|(1u<<6)))return false;
#endif
    for(auto v:{q.r0,q.c0,q.r1,q.c1,q.r2,q.c2})if(v < -(Int(1)<<20)||v > (Int(1)<<20))return false;
    if(!std::isfinite(q.weight)||q.weight<0||q.weight>16)return false;
    if(q.clipped && (!std::isfinite(q.height)||!std::isfinite(q.width)||std::abs(q.height)>0x1p52||std::abs(q.width)>0x1p52))return false;
    Environment environment;
    BezierCurve p;p.request=q;Builder builder{p};
    if(!builder.curve() || std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW))return false;
    BEZIER_TRACE("complete visited=%zu emitted=%zu\n",builder.visited,p.rows.size());
    output=std::move(p);return true;
}

void bezier_curve_fill(const BezierCurve& p,DrawCoordinatePair out,int workers) noexcept {
    const auto n=p.rows.size();
    const auto* rows=p.rows.data();const auto* columns=p.columns.data();
    auto* output_rows=out.rows;auto* output_columns=out.columns;
    const auto row_step=out.row_step,column_step=out.column_step;
    partition(n,n>=parallel_min/2?workers:1,[=](std::size_t begin,std::size_t end,std::size_t) {
        if(row_step==1 && column_step==1) {
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd
#endif
            for(auto i=begin;i<end;++i) {
                output_rows[i]=rows[i];output_columns[i]=columns[i];
            }
            return;
        }
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd
#endif
        for(auto i=begin;i<end;++i) {
            output_rows[static_cast<std::ptrdiff_t>(i)*row_step]=rows[i];
            output_columns[static_cast<std::ptrdiff_t>(i)*column_step]=columns[i];
        }
    });
}
}
