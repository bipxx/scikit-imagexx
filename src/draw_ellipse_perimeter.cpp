// Literal ordered recurrences from scikit-image0.26.0 _draw.pyx.
// BSD-3-Clause: third_party/scikit-image-LICENSE.txt.
#include <skimagexx/draw_ellipse_perimeter.hpp>
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace skimagexx::detail {
namespace {
using Int=std::intptr_t;
// Restricted source primitive: only the four checked unit-circle half-weight
// corner segments call this. General rational subdivision is not implemented.
template<class Emit> bool unit_segment(Int r0,Int c0,Int r1,Int c1,Int r2,Int c2,double weight,Emit emit) {
    double sc=double(c2-c1),sr=double(r2-r1);
    const double d2c0=double(c0-c2),d2r0=double(r0-r2);
    double d2c=d2c0,d2r=d2r0,d1c=double(c0-c1),d1r=double(r0-r1);
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
        if(weight<.5 && (d2r>rc || d2c<rc))return false;
        double err=d2c+d2r-rc;
        unsigned iterations=0;
        while(d2r<=rc && d2c>=rc) {
            if(++iterations>1)return false; // Proven unit-domain bound, not a generic curve quota.
            emit(r0,c0);
            if(c0==c2 && r0==r2)return true;
            const bool test1=2*err>d2r,test2=2*(err+d1r)<-d2r;
            if(2*err<d2c || test2) {r0+=static_cast<Int>(sr);d2r+=rc;d2c+=d1c;err+=d2c;}
            if(2*err>d2c || test1) {c0+=static_cast<Int>(sc);d2c+=rc;d2r+=d1r;err+=d2r;}
        }
    }
    // In this admitted domain the source's _line fallback is endpoint->endpoint.
    // Emit that one source line point; do not substitute a general Bezier/line.
    if(c0!=c2 || r0!=r2)return false;
    emit(r0,c0);return true;
}
}

bool prepare_ellipse_perimeter(const EllipsePerimeterRequest& q,EllipsePerimeter& output) {
    if(sizeof(Int)!=8 || q.row_radius<0 || q.column_radius<0 ||
       q.row_radius>32768 || q.column_radius>32768)return false;
    EllipsePerimeter p;p.request=q;
    if(q.row_radius==0 && q.column_radius==0) {
        p.rows.push_back(q.row);p.columns.push_back(q.column);p.scalar_result=true;
        output=std::move(p);return true;
    }
    // The source's both-zero return precedes all orientation math. Conversion
    // to double belongs to the frontend; even NaN/Inf are ignored by that point.
    if(!std::isfinite(q.orientation))return false;
    if(q.clipped && (!std::isfinite(q.height)||!std::isfinite(q.width)||
       std::abs(q.height)>0x1p52||std::abs(q.width)>0x1p52))return false;
    const bool rotated=q.orientation!=0;
    if(rotated && (q.row!=0 || q.column!=0 || q.row_radius!=1 || q.column_radius!=1 || std::abs(q.orientation)>.5))return false;
    const Int row_extent=q.row_radius+3,column_extent=q.column_radius+1;
    constexpr auto lo=std::numeric_limits<Int>::min(),hi=std::numeric_limits<Int>::max();
    if(q.row<lo+row_extent||q.row>hi-row_extent||q.column<lo+column_extent||q.column>hi-column_extent)return false;
    const auto maximum=std::size_t(4)*(std::size_t(q.row_radius)+std::size_t(q.column_radius)+3)+2*std::size_t(q.row_radius);
    if(maximum>1000000)return false;
    p.rows.reserve(maximum);p.columns.reserve(maximum);
    const auto emit=[&](Int row,Int column) {
        row+=q.row;column+=q.column;
        if(!q.clipped || (row>=0 && column>=0 && double(row)<q.height && double(column)<q.width)) {
            p.rows.push_back(row);p.columns.push_back(column);
        }
    };
    const Int rd=q.row_radius*q.row_radius,cd=q.column_radius*q.column_radius;
    if(!rotated) {
        Int c=-q.column_radius,r=0,e2=rd,err=c*(2*e2+c)+e2;
        std::size_t iterations=0;
        while(c<=0) {
            if(r>q.row_radius+2 || ++iterations>std::size_t(q.row_radius+q.column_radius+3))return false;
            emit(r,-c);emit(r,c);emit(-r,c);emit(-r,-c);
            const auto old_r=r,old_c=c;
            e2=2*err;
            if(e2>=(2*c+1)*rd) {++c;err+=(2*c+1)*rd;}
            if(e2<=(2*r+1)*cd) {++r;err+=(2*r+1)*cd;}
            if(r==old_r && c==old_c)return false;
        }
        while(r<q.row_radius) {++r;emit(r,0);emit(-r,0);}
    } else {
        // Preserve literal source operation association and C-int corner stage.
        const double sin_angle=std::sin(q.orientation);
        double za=double(cd-rd)*sin_angle;
        double ca=std::sqrt(double(cd)-za*sin_angle),ra=std::sqrt(double(rd)+za*sin_angle);
        const double a=ca+.5,b=ra+.5;
        za=za*a*b/(ca*ra);
        const int ir0=static_cast<int>(double(q.row)-b),ic0=static_cast<int>(double(q.column)-a);
        const int ir1=static_cast<int>(double(q.row)+b),ic1=static_cast<int>(double(q.column)+a);
        ca=double(ic1-ic0);ra=double(ir1-ir0);za=4*za*std::cos(q.orientation);
        double w=ca*ra;if(w!=0)w=(w-za)/(w+w);
        const int icd=static_cast<int>(std::floor(ca*w+.5)),ird=static_cast<int>(std::floor(ra*w+.5));
        if(!unit_segment(ir0+ird,ic0,ir0,ic0,ir0,ic0+icd,1-w,emit) ||
           !unit_segment(ir0+ird,ic0,ir1,ic0,ir1,ic1-icd,w,emit) ||
           !unit_segment(ir1-ird,ic1,ir1,ic1,ir1,ic1-icd,1-w,emit) ||
           !unit_segment(ir1-ird,ic1,ir0,ic1,ir0,ic0+icd,w,emit))return false;
    }
    output=std::move(p);return true;
}

void ellipse_perimeter_fill(const EllipsePerimeter& p,DrawCoordinatePair out,int workers) noexcept {
    const auto size=p.rows.size();const Int* rows=p.rows.data();const Int* columns=p.columns.data();
    const int active=size>=parallel_min/2?workers:1;
    partition(size,active,[&](std::size_t begin,std::size_t end,std::size_t) {
        if(out.row_step==1 && out.column_step==1) {
            #if defined(_OPENMP) && !defined(_MSC_VER)
            #pragma omp simd
            #endif
            for(auto i=begin;i<end;++i) {out.rows[i]=rows[i];out.columns[i]=columns[i];}
        } else {
            for(auto i=begin;i<end;++i) {
                out.rows[std::ptrdiff_t(i)*out.row_step]=rows[i];
                out.columns[std::ptrdiff_t(i)*out.column_step]=columns[i];
            }
        }
    });
}
}
