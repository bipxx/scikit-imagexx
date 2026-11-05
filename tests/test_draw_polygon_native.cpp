#include <skimagexx/draw_polygon.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t controls=0,records=0;
void check(bool yes){++controls;if(!yes)throw std::runtime_error("draw polygon native control failed");}
struct Original {const char* id;std::vector<double> r,c;DrawPolygonScan scan;std::size_t rows,columns;std::vector<std::intptr_t> rr,cc;std::vector<std::uint8_t> mask;};
// Exact decoded pre-candidate originals; mask->coords or coords->mask is only
// an output representation projection, never a geometric third-formula oracle.
const std::vector<Original> originals={
{"draw.polygon/triangle-list",{0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+2},{0x1.0000000000000p+0,0x1.4000000000000p+2,0x1.0000000000000p+1},{{1,1},{4,5}},5,6,{1,1,1,1,1,2,2,2,3,3,4},{1,2,3,4,5,2,3,4,2,3,2},{0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1,0,0,0}},
{"draw.polygon/rectangle-i32",{0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x1.8000000000000p+1},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{{0,0},{3,4}},6,7,{0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3},{0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4},{1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
{"draw.polygon/concave-reverse",{0x1.4000000000000p+2,0x1.4000000000000p+2,0x1.0000000000000p+1,0x0.0p+0,0x0.0p+0},{0x0.0p+0,0x1.4000000000000p+2,0x1.0000000000000p+1,0x1.4000000000000p+2,0x0.0p+0},{{0,0},{5,5}},6,6,{0,0,0,0,0,0,1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5},{0,1,2,3,4,5,0,1,2,3,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5},{1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,0,0,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,1}},
{"draw.polygon/bowtie-strided",{0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{{0,0},{4,4}},6,6,{0,0,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4},{0,4,0,1,3,4,0,1,2,3,4,0,1,3,4,0,4},{1,0,0,0,1,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0}},
{"draw.polygon/negative-clip",{-0x1.0000000000000p+1,0x1.0000000000000p+0,0x1.4000000000000p+2},{-0x1.8000000000000p+1,0x1.4000000000000p+2,0x1.0000000000000p+0},{{0,0},{4,5}},5,6,{0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4},{0,1,2,0,1,2,3,4,5,0,1,2,3,4,0,1,2,3,1,2},{1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1,1,0,0,0}},
{"draw.polygon/fully-negative",{-0x1.0000000000000p+2,-0x1.0000000000000p+1,-0x1.0000000000000p+1},{-0x1.8000000000000p+1,-0x1.8000000000000p+1,-0x1.0000000000000p+0},{{0,0},{-2,-1}},0,0,{},{},{}},
{"draw.polygon/zero-shape",{0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x1.8000000000000p+1},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{{0,0},{-1,4}},0,6,{},{},{}},
{"draw.polygon/epsilon-vertex",{0x1.0000000000000p-41,0x0.0p+0,0x1.8000000000000p+1},{0x1.0000000000000p-41,0x1.8000000000000p+1,0x0.0p+0},{{0,0},{3,3}},5,5,{0,0,1,1,2,3},{0,3,1,2,1,0},{1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0}},
{"draw.polygon/singleton",{0x1.0000000000000p+1},{0x1.8000000000000p+1},{{2,3},{2,3}},3,4,{2},{3},{0,0,0,0,0,0,0,0,0,0,0,1}},
{"draw.polygon/collinear-duplicate",{0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+0},{0x0.0p+0,0x1.0000000000000p+1,0x1.0000000000000p+1,0x1.0000000000000p+2},{{1,0},{1,4}},5,6,{1,1,1},{0,2,4},{0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
{"draw.polygon2mask/triangle-list",{0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+2},{0x1.0000000000000p+0,0x1.4000000000000p+2,0x1.0000000000000p+1},{{1,1},{4,5}},6,7,{1,1,1,1,1,2,2,2,3,3,4},{1,2,3,4,5,2,3,4,2,3,2},{0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}},
{"draw.polygon2mask/concave-f32-C",{0x0.0p+0,0x0.0p+0,0x1.0000000000000p+1,0x1.4000000000000p+2,0x1.4000000000000p+2},{0x0.0p+0,0x1.4000000000000p+2,0x1.0000000000000p+1,0x1.4000000000000p+2,0x0.0p+0},{{0,0},{5,5}},7,8,{0,0,0,0,0,0,1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5},{0,1,2,3,4,5,0,1,2,3,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5},{1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}},
{"draw.polygon2mask/negative-strided",{-0x1.0000000000000p+1,0x1.0000000000000p+0,0x1.4000000000000p+2},{-0x1.8000000000000p+1,0x1.4000000000000p+2,0x1.0000000000000p+0},{{0,0},{4,5}},5,6,{0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4},{0,1,2,0,1,2,3,4,5,0,1,2,3,4,0,1,2,3,1,2},{1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1,1,0,0,0}},
{"draw.polygon2mask/outside",{0x1.0000000000000p+3,0x1.0000000000000p+3,0x1.4000000000000p+3},{0x1.0000000000000p+3,0x1.4000000000000p+3,0x1.0000000000000p+3},{{8,8},{2,3}},3,4,{},{},{0,0,0,0,0,0,0,0,0,0,0,0}},
{"draw.polygon2mask/zero-shape",{0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x1.8000000000000p+1},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{{0,0},{-1,4}},0,6,{},{},{}},
{"draw.polygon2mask/epsilon-vertex",{0x1.0000000000000p-41,0x0.0p+0,0x1.8000000000000p+1},{0x1.0000000000000p-41,0x1.8000000000000p+1,0x0.0p+0},{{0,0},{3,3}},4,4,{0,0,1,1,2,3},{0,3,1,2,1,0},{1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,0}},
{"draw.polygon2mask/singleton",{0x1.0000000000000p+1},{0x1.8000000000000p+1},{{2,3},{2,3}},4,5,{2},{3},{0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}},
};
void same_plan(const DrawPolygonPlan& a,const DrawPolygonPlan& b) {
    check(a.first==b.first&&a.shape==b.shape&&a.labels==b.labels&&a.row_offsets==b.row_offsets&&a.count==b.count);
}
void saved(int workers) {
    for(const auto& spec:originals) {
        std::cout<<"saved "<<spec.id<<" worker="<<workers<<'\n';
        const auto r_before=spec.r,c_before=spec.c;
        DrawPolygonPlan plan;
        check(draw_polygon_prepare({spec.r.data(),spec.c.data(),spec.r.size()},spec.scan,plan,workers));
        check(plan.count==spec.rr.size()&&spec.rr.size()==spec.cc.size());
        check(plan.labels.size()==plan.shape[0]*plan.shape[1]&&plan.row_offsets.size()==plan.shape[0]+1);
        std::size_t selected=0;for(auto label:plan.labels){check(label<=3);selected+=label!=0;}
        check(selected==plan.count&&plan.row_offsets.back()==plan.count);
        const auto copy=plan;
        std::vector<std::intptr_t> rr(plan.count+2,-999),cc(plan.count+2,-999);
        draw_polygon_coords_into(plan,rr.data()+1,cc.data()+1,workers);
        check(rr.front()==-999&&rr.back()==-999&&cc.front()==-999&&cc.back()==-999);
        check(std::equal(spec.rr.begin(),spec.rr.end(),rr.begin()+1));
        check(std::equal(spec.cc.begin(),spec.cc.end(),cc.begin()+1));
        std::vector<std::uint8_t> mask(spec.mask.size()+2,91);
        draw_polygon_mask_into(plan,spec.rows,spec.columns,mask.data()+1,workers);
        check(mask.front()==91&&mask.back()==91&&std::equal(spec.mask.begin(),spec.mask.end(),mask.begin()+1));
        same_plan(plan,copy);check(spec.r==r_before&&spec.c==c_before);++records;
    }
}
void analytic(int workers) {
    const std::array<double,4> r{0,0,2,2},c{0,2,2,0};
    DrawPolygonPlan plan;check(draw_polygon_prepare({r.data(),c.data(),4},{{0,0},{3,3}},plan,workers));
    check(plan.count==9&&plan.labels==std::vector<std::uint8_t>({2,3,2,0,3,1,3,0,2,3,2,0,0,0,0,0}));
    std::array<std::uint8_t,11> mask;mask.fill(91);
    // Only selected points must fit: the raster's final row/column are outside.
    draw_polygon_mask_into(plan,3,3,mask.data()+1,workers);
    check(mask.front()==91&&mask.back()==91&&std::count(mask.begin()+1,mask.end()-1,std::uint8_t(1))==9);
    const auto before=mask;bool refused=false;
    try{draw_polygon_mask_into(plan,2,2,mask.data()+1,workers);}catch(const std::invalid_argument&){refused=true;}
    check(refused&&mask==before);
    refused=false;try{draw_polygon_mask_into(plan,std::numeric_limits<std::size_t>::max(),2,mask.data()+1,workers);}catch(const std::invalid_argument&){refused=true;}
    check(refused&&mask==before);
    for(int fault=0;fault<4;++fault) {
        refused=false;
        try{
            if(fault==0)draw_polygon_coords_into(plan,nullptr,nullptr,workers);
            if(fault==1)draw_polygon_coords_into(plan,nullptr,nullptr,0);
            if(fault==2)draw_polygon_mask_into(plan,3,3,nullptr,workers);
            if(fault==3)draw_polygon_mask_into(plan,3,3,mask.data()+1,0);
        }catch(const std::invalid_argument&){refused=true;}
        check(refused&&mask==before);
    }
    DrawPolygonPlan empty;check(draw_polygon_prepare({r.data(),c.data(),4},{{0,0},{-1,2}},empty,workers));
    check(empty.count==0&&empty.labels.empty()&&empty.row_offsets==std::vector<std::size_t>{0});
    draw_polygon_coords_into(empty,nullptr,nullptr,workers);
    draw_polygon_mask_into(empty,0,0,nullptr,workers);
    std::array<std::uint8_t,8> zeros;zeros.fill(91);
    draw_polygon_mask_into(empty,2,3,zeros.data()+1,workers);
    check(zeros.front()==91&&zeros.back()==91&&std::count(zeros.begin()+1,zeros.end()-1,std::uint8_t(0))==6);
    // Source-derived integer square, not an additional original capture.
    const std::array<double,4> lr{0,0,512,512},lc{0,512,512,0};
    DrawPolygonPlan large;check(draw_polygon_prepare({lr.data(),lc.data(),4},{{0,0},{512,512}},large,workers));
    constexpr std::size_t width=513,n=width*width;check(large.count==n);
    std::vector<std::intptr_t> rr(n+2,-999),cc(n+2,-999);
    std::vector<std::uint8_t> image(n+2,91);
    draw_polygon_coords_into(large,rr.data()+1,cc.data()+1,workers);
    draw_polygon_mask_into(large,width,width,image.data()+1,workers);
    for(std::size_t i=0;i<n;++i){check(rr[i+1]==std::intptr_t(i/width));check(cc[i+1]==std::intptr_t(i%width));check(image[i+1]==1);}
    check(rr.front()==-999&&rr.back()==-999&&cc.front()==-999&&cc.back()==-999&&image.front()==91&&image.back()==91);
}
void refusals(int workers) {
    const std::array<double,4> r{0,0,2,2},c{0,2,2,0};DrawPolygonPlan guard;
    check(draw_polygon_prepare({r.data(),c.data(),4},{{0,0},{2,2}},guard,workers));
    for(int fault=0;fault<17;++fault) {
        auto vr=r,vc=c;DrawPolygonVertices view{vr.data(),vc.data(),4};DrawPolygonScan scan{{0,0},{2,2}};
        std::vector<double> many(4096,0);int budget=workers;auto result=guard;
        const auto oldround=std::fegetround();
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        const auto csr=_mm_getcsr();
#else
        if(fault==16)continue;
#endif
        if(fault==0)view.count=0;
        if(fault==1)view.rows=nullptr;
        if(fault==2)view.columns=nullptr;
        if(fault==3)view.count=draw_polygon_max_vertices+1;
        if(fault==4)vr[0]=std::numeric_limits<double>::quiet_NaN();
        if(fault==5)vc[0]=std::numeric_limits<double>::infinity();
        if(fault==6)vr[0]=0x1p21;
        if(fault==7)vr[0]=0x1p-129;
        if(fault==8)scan.first[0]=-1;
        if(fault==9)scan.first[0]=(1<<20)+1;
        if(fault==10)scan.last[1]=(1<<20)+1;
        if(fault==11)scan.last={4096,4096};
        if(fault==12){view={many.data(),many.data(),many.size()};scan.last={128,128};}
        if(fault==13)budget=0;
        if(fault==14)budget=1025;
        if(fault==15)check(std::fesetround(FE_DOWNWARD)==0);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        if(fault==16)_mm_setcsr(csr|(1u<<15)|(1u<<6));
#endif
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);check(!draw_polygon_prepare(view,scan,result,budget));
        check(std::fetestexcept(FE_ALL_EXCEPT)==flags);same_plan(guard,result);
        check(std::fesetround(oldround)==0);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        _mm_setcsr(csr);
#endif
    }
}
void mxcsr_boundary(int workers) {
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    check(std::fegetround()==FE_TONEAREST);
    const auto csr=_mm_getcsr(),changed=(csr&~(3u<<13))|(1u<<13);
    _mm_setcsr(changed);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    const auto reported=std::fegetround();
    check(reported==FE_TONEAREST||reported==FE_DOWNWARD);
    const std::array<double,4> r{0,0,2,2},c{0,2,2,0};
    DrawPolygonPlan plan;plan.first={17,19};plan.row_offsets={0};const auto before=plan;
    check(!draw_polygon_prepare({r.data(),c.data(),4},{{0,0},{2,2}},plan,workers));
    check(_mm_getcsr()==changed&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
    same_plan(before,plan);_mm_setcsr(csr);
#else
    (void)workers;
#endif
}
}
int main(int argc,char** argv) {
    check(argc==1||(argc==2&&std::string(argv[1])=="--mxcsr-only"));
    std::fenv_t old;std::fegetenv(&old);check(std::fesetround(FE_TONEAREST)==0);
    const bool available=has_avx2();check(get_num_threads()==1);
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    for(int workers:{1
#ifdef _OPENMP
        ,4
#endif
    }) {
        set_num_threads(workers);const auto snapshot=get_num_threads();
        if(argc==1){saved(snapshot);analytic(snapshot);refusals(snapshot);}
        mxcsr_boundary(snapshot);
        check(get_num_threads()==workers&&std::fegetround()==FE_TONEAREST&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
    }
    set_num_threads(1);check(get_num_threads()==1);std::fesetenv(&old);
    std::cout<<"draw polygon saved records="<<records<<"; controls="<<controls<<"; default=1; available_AVX2="<<available<<'\n';
}
