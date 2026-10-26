#include "skimagexx/footprint_shapes.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cfenv>
#pragma STDC FENV_ACCESS ON

namespace skimagexx { namespace {
constexpr std::size_t limit=std::size_t(1)<<20;
std::size_t product(std::size_t a,std::size_t b){
    if(b && a>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/b)
        throw std::overflow_error("footprint span overflows");
    return a*b;
}
struct Range{std::uintptr_t a,b;};
Range range(const void* p,std::size_t n){
    const auto a=reinterpret_cast<std::uintptr_t>(p);
    if((n&&!p)||n>std::numeric_limits<std::uintptr_t>::max()-a)
        throw std::invalid_argument("invalid footprint byte span");
    return {a,a+n};
}
void out(FootprintBytes value,std::size_t bytes){
    if(value.bytes!=bytes)throw std::invalid_argument("footprint exact output size required");
    range(value.data,value.bytes);
}
void disjoint(FootprintBytes a,FootprintConstBytes b){
    const auto x=range(a.data,a.bytes),y=range(b.data,b.bytes);
    if(x.a<x.b && y.a<y.b && x.a<y.b && y.a<x.b)
        throw std::invalid_argument("footprint spans overlap");
}
void dimensions(std::size_t r,std::size_t c){
    if(!r||!c||r>limit||c>limit)throw std::invalid_argument("footprint finite dimensions unsupported");
}
struct Point{double r,c;};
bool operator<(Point a,Point b){return a.r<b.r||(a.r==b.r&&a.c<b.c);}
bool operator==(Point a,Point b){return a.r==b.r&&a.c==b.c;}
double cross(Point a,Point b,Point c){return (b.r-a.r)*(c.c-a.c)-(b.c-a.c)*(c.r-a.r);}
bool inside(const std::vector<Point>& hull,double r,double c){
    // Source _shared/geometry.pyx ray classification, including vertex/edge
    // tolerance. scikit-image BSD license retained in repository LICENSE.
    unsigned left=0,right=0;
    double x1=hull.back().r-r,y1=hull.back().c-c;
    for(const auto& p:hull){
        const double x0=p.r-r,y0=p.c-c;
        if(-1e-12<x0&&x0<1e-12&&-1e-12<y0&&y0<1e-12)return true;
        if((y0>0)!=(y1>0))if((x0*y1-x1*y0)/(y1-y0)>0)++right;
        if((y0<0)!=(y1<0))if((x0*y1-x1*y0)/(y1-y0)<0)++left;
        x1=x0;y1=y0;
    }
    return (left&1)!=(right&1)||(right&1);
}
struct Fenv{std::fenv_t saved;Fenv(){if(std::feholdexcept(&saved))throw std::runtime_error("cannot preserve footprint FEnv");}~Fenv(){std::fesetenv(&saved);}};
}
void footprint_shapes_ellipse(FootprintBytes target,std::size_t rows,std::size_t columns,double height,double width){
    dimensions(rows,columns);const auto n=product(rows,columns);out(target,n);
    if(!std::isfinite(height)||!std::isfinite(width)||height<0||width<0||
       height>double(limit)||width>double(limit)||std::fegetround()!=FE_TONEAREST)
        throw std::invalid_argument("unsupported ellipse geometry");
    Fenv keep;
    for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<columns;++c){
            // Literal rotation0 draw expression; preserve two divisions and
            // binary64 squaring/add association (compile strict, no FMA).
            const double dr=double(r)-height,dc=double(c)-width;
            const double a=(dr*1.0+dc*0.0)/(height+1.0);
            const double b=(dr*0.0-dc*1.0)/(width+1.0);
            static_cast<unsigned char*>(target.data)[r*columns+c]=(a*a+b*b)<1.0;
    }
}
void footprint_shapes_hull(FootprintConstBytes input,FootprintBytes target,std::size_t rows,std::size_t columns){
    dimensions(rows,columns);const auto n=product(rows,columns);
    if(input.bytes!=n)throw std::invalid_argument("footprint hull input size mismatch");
    range(input.data,input.bytes);out(target,n);disjoint(target,input);
    std::vector<Point> points;
    const auto* bytes=static_cast<const unsigned char*>(input.data);
    // Only row/column extrema can change a convex hull. This is the same
    // source possible_hull optimization; repetitions are removed by sort.
    auto append=[&](std::size_t r,std::size_t c){
        points.push_back({double(r)-.5,double(c)});points.push_back({double(r)+.5,double(c)});
        points.push_back({double(r),double(c)-.5});points.push_back({double(r),double(c)+.5});
    };
    for(std::size_t r=0;r<rows;++r){
        std::size_t first=columns,last=columns;
        for(std::size_t c=0;c<columns;++c)if(bytes[r*columns+c]){if(first==columns)first=c;last=c;}
        if(first!=columns){append(r,first);if(last!=first)append(r,last);}
    }
    for(std::size_t c=0;c<columns;++c){
        std::size_t first=rows,last=rows;
        for(std::size_t r=0;r<rows;++r)if(bytes[r*columns+c]){if(first==rows)first=r;last=r;}
        if(first!=rows){append(first,c);if(last!=first)append(last,c);}
    }
    if(points.empty())throw std::invalid_argument("footprint hull requires nonzero pixels");
    std::sort(points.begin(),points.end());points.erase(std::unique(points.begin(),points.end()),points.end());
    std::vector<Point> hull;
    for(auto p:points){while(hull.size()>1&&cross(hull[hull.size()-2],hull.back(),p)<=0)hull.pop_back();hull.push_back(p);}
    const auto lower=hull.size();
    for(auto i=points.size()-1;i-->0;){auto p=points[i];while(hull.size()>lower&&cross(hull[hull.size()-2],hull.back(),p)<=0)hull.pop_back();hull.push_back(p);}
    hull.pop_back();if(hull.size()<3)throw std::invalid_argument("degenerate footprint hull");
    Fenv keep;for(std::size_t r=0;r<rows;++r)
        for(std::size_t c=0;c<columns;++c)static_cast<unsigned char*>(target.data)[r*columns+c]=inside(hull,double(r),double(c));
}
void footprint_shapes_l1(FootprintConstBytes x,FootprintConstBytes y,FootprintConstBytes z,FootprintBytes target,std::size_t n,double radius,bool inclusive){
    const auto bytes=product(n,sizeof(double));out(target,n);
    if(!std::isfinite(radius)||std::fegetround()!=FE_TONEAREST)throw std::invalid_argument("unsupported octahedron geometry");
    for(auto source:{x,y,z}){if(source.bytes!=bytes)throw std::invalid_argument("octahedron grid size mismatch");range(source.data,bytes);disjoint(target,source);}
    Fenv keep;for(std::size_t i=0;i<n;++i){
        double a,b,c;std::memcpy(&a,static_cast<const char*>(x.data)+8*i,8);std::memcpy(&b,static_cast<const char*>(y.data)+8*i,8);std::memcpy(&c,static_cast<const char*>(z.data)+8*i,8);
        const double sum=(std::abs(a)+std::abs(b))+std::abs(c);
        static_cast<unsigned char*>(target.data)[i]=inclusive?sum<=radius:sum<radius;
    }
}
std::size_t footprint_shapes_octahedron_axis(std::size_t radius){
    const auto maximum=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(radius>(maximum-1)/2)throw std::overflow_error("octahedron axis overflows");
    const auto n=2*radius+1;
    const auto count=product(product(n,n),n);
    product(count,3*sizeof(double)); // preserve original dense mgrid addressability
    return n;
}
void footprint_shapes_octahedron(FootprintBytes target,std::size_t radius){
    const auto n=footprint_shapes_octahedron_axis(radius),plane=n*n,count=plane*n;
    out(target,count);
    auto* data=static_cast<unsigned char*>(target.data);
    detail::partition(n,detail::active_workers(count),[&](auto begin,auto end,auto){
        for(auto z=begin;z<end;++z){
            auto* slice=data+z*plane;std::memset(slice,0,plane);
            const auto dz=z>radius?z-radius:radius-z;
            for(std::size_t y=0;y<n;++y){
                const auto dy=y>radius?y-radius:radius-y;
                if(dz+dy<=radius){
                    const auto remaining=radius-dz-dy;
                    std::memset(slice+y*n+radius-remaining,1,2*remaining+1);
                }
            }
        }
    });
}
void footprint_shapes_atoms(FootprintConstBytes mask,FootprintBytes target,FootprintConstBytes zero,FootprintConstBytes one,std::size_t n,std::size_t item){
    if(!item||zero.bytes!=item||one.bytes!=item||mask.bytes!=n)throw std::invalid_argument("footprint atoms size mismatch");
    out(target,product(n,item));for(auto source:{mask,zero,one}){range(source.data,source.bytes);disjoint(target,source);}
    const auto workers=detail::active_workers(n);
    detail::partition(n,workers,[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)
        std::memcpy(static_cast<char*>(target.data)+i*item,static_cast<const unsigned char*>(mask.data)[i]?one.data:zero.data,item);});
}
}
