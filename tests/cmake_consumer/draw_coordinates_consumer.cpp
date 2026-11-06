// Installed SDK consumer only: no private source headers, Python or originals.
#include <skimagexx/draw_coordinates.hpp>
#include <skimagexx/image.hpp>
#include <skimagexx/line_aa.hpp>
#include <skimagexx/circle_perimeter_aa.hpp>
#include <skimagexx/draw_polygon.hpp>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
#include <cstring>
#include <limits>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0;
void check(bool value){++checks;if(!value)throw std::runtime_error("installed drawing coordinate mismatch "+std::to_string(checks));}
// Immutable saved words; mask/coords conversion is representation projection.
void polygon(int workers){
    const auto global=get_num_threads();std::fenv_t prior;check(std::fegetenv(&prior)==0);
    const auto rounding=std::fegetround(),flags=std::fetestexcept(FE_ALL_EXCEPT);
    struct Saved{const char* id;std::vector<double> r,c;DrawPolygonScan scan;std::size_t height,width;std::vector<std::intptr_t> rr,cc;std::vector<std::uint8_t> mask;};
    const std::vector<Saved> saved={
{"draw.polygon/triangle-list",{0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+2},{0x1.0000000000000p+0,0x1.4000000000000p+2,0x1.0000000000000p+1},{{1,1},{4,5}},5,6,{1,1,1,1,1,2,2,2,3,3,4},{1,2,3,4,5,2,3,4,2,3,2},{0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1,0,0,0}},
{"draw.polygon/bowtie-strided",{0x0.0p+0,0x1.0000000000000p+2,0x0.0p+0,0x1.0000000000000p+2},{0x0.0p+0,0x1.0000000000000p+2,0x1.0000000000000p+2,0x0.0p+0},{{0,0},{4,4}},6,6,{0,0,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4},{0,4,0,1,3,4,0,1,2,3,4,0,1,3,4,0,4},{1,0,0,0,1,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0}},
{"draw.polygon/epsilon-vertex",{0x1.0000000000000p-41,0x0.0p+0,0x1.8000000000000p+1},{0x1.0000000000000p-41,0x1.8000000000000p+1,0x0.0p+0},{{0,0},{3,3}},5,5,{0,0,1,1,2,3},{0,3,1,2,1,0},{1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0}},
{"draw.polygon/fully-negative",{-0x1.0000000000000p+2,-0x1.0000000000000p+1,-0x1.0000000000000p+1},{-0x1.8000000000000p+1,-0x1.8000000000000p+1,-0x1.0000000000000p+0},{{0,0},{-2,-1}},0,0,{},{},{}},
{"draw.polygon2mask/triangle-list",{0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p+2},{0x1.0000000000000p+0,0x1.4000000000000p+2,0x1.0000000000000p+1},{{1,1},{4,5}},6,7,{1,1,1,1,1,2,2,2,3,3,4},{1,2,3,4,5,2,3,4,2,3,2},{0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}},
{"draw.polygon2mask/outside",{0x1.0000000000000p+3,0x1.0000000000000p+3,0x1.4000000000000p+3},{0x1.0000000000000p+3,0x1.4000000000000p+3,0x1.0000000000000p+3},{{8,8},{2,3}},3,4,{},{},{0,0,0,0,0,0,0,0,0,0,0,0}},
    };
    for(const auto& spec:saved){
        DrawPolygonPlan plan;check(draw_polygon_prepare({spec.r.data(),spec.c.data(),spec.r.size()},spec.scan,plan,workers));
        check(plan.count==spec.rr.size()&&plan.count==spec.cc.size());
        const auto labels=plan.labels;const auto prefix=plan.row_offsets;
        std::vector<std::intptr_t> rr(plan.count+2,-999),cc(plan.count+2,-999);
        draw_polygon_coords_into(plan,rr.data()+1,cc.data()+1,workers);
        check(rr.front()==-999&&rr.back()==-999&&cc.front()==-999&&cc.back()==-999);
        check(std::equal(spec.rr.begin(),spec.rr.end(),rr.begin()+1)&&std::equal(spec.cc.begin(),spec.cc.end(),cc.begin()+1));
        std::vector<std::uint8_t> mask(spec.mask.size()+2,91);draw_polygon_mask_into(plan,spec.height,spec.width,mask.data()+1,workers);
        check(mask.front()==91&&mask.back()==91&&std::equal(spec.mask.begin(),spec.mask.end(),mask.begin()+1));
        check(plan.labels==labels&&plan.row_offsets==prefix&&plan.row_offsets.back()==plan.count);
        check(std::all_of(mask.begin()+1,mask.end()-1,[](auto v){return v<=1;}));
        if(!plan.count){draw_polygon_coords_into(plan,nullptr,nullptr,workers);draw_polygon_mask_into(plan,0,0,nullptr,workers);}
    }
    // Inclusive rectangles: analytic row/column order and independent large spans.
    for(std::size_t size:{std::size_t(3),std::size_t(513)}){
        const double edge=double(size-1);std::array<double,4> r{0,0,edge,edge},c{0,edge,edge,0};
        DrawPolygonPlan plan;check(draw_polygon_prepare({r.data(),c.data(),4},{{0,0},{std::intptr_t(size-1),std::intptr_t(size-1)}},plan,workers));
        check(plan.count==size*size);
        r.fill(-9);c.fill(-9); // Prepared plan owns geometry; no borrowed vertex storage.
        std::vector<std::intptr_t> rr(plan.count+2,-999),cc(plan.count+2,-999);
        draw_polygon_coords_into(plan,rr.data()+1,cc.data()+1,workers);
        std::vector<std::uint8_t> mask(plan.count+2,91);draw_polygon_mask_into(plan,size,size,mask.data()+1,workers);
        for(std::size_t i=0;i<plan.count;++i)check(rr[i+1]==std::intptr_t(i/size)&&cc[i+1]==std::intptr_t(i%size)&&mask[i+1]==1);
        check(rr.front()==-999&&rr.back()==-999&&cc.front()==-999&&cc.back()==-999&&mask.front()==91&&mask.back()==91);
    }
    const std::array<double,4> r{0,0,2,2},c{0,2,2,0};DrawPolygonPlan plan;
    check(draw_polygon_prepare({r.data(),c.data(),4},{{0,0},{2,2}},plan,workers));
    const auto labels=plan.labels;const auto offsets=plan.row_offsets;const auto count=plan.count;const auto first=plan.first;const auto shape=plan.shape;
    std::array<std::uint8_t,11> mask;mask.fill(91);const auto before=mask;
    for(int fault=0;fault<4;++fault){
        bool refused=false;
        try{
            if(fault==0)draw_polygon_mask_into(plan,2,2,mask.data()+1,workers);
            if(fault==1)draw_polygon_mask_into(plan,std::numeric_limits<std::size_t>::max(),2,mask.data()+1,workers);
            if(fault==2)draw_polygon_mask_into(plan,3,3,nullptr,workers);
            if(fault==3)draw_polygon_mask_into(plan,3,3,mask.data()+1,0);
        }catch(const std::invalid_argument&){refused=true;}
        check(refused&&mask==before);
    }
    bool refused=false;try{draw_polygon_coords_into(plan,nullptr,nullptr,workers);}catch(const std::invalid_argument&){refused=true;}check(refused);
    for(int fault=0;fault<8;++fault){
        auto bad=r;DrawPolygonVertices vertices{bad.data(),c.data(),4};DrawPolygonScan scan{{0,0},{2,2}};
        if(fault==0)vertices.count=0;
        if(fault==1)vertices.rows=nullptr;
        if(fault==2)bad[0]=std::numeric_limits<double>::infinity();
        if(fault==3)bad[0]=std::numeric_limits<double>::quiet_NaN();
        if(fault==4)scan.first[0]=-1;
        if(fault==5)bad[0]=std::numeric_limits<double>::denorm_min();
        if(fault==6)bad[0]=double(1<<21);
        if(fault==7)vertices.count=draw_polygon_max_vertices+1;
        check(!draw_polygon_prepare(vertices,scan,plan,workers));check(plan.labels==labels&&plan.row_offsets==offsets&&plan.count==count&&plan.first==first&&plan.shape==shape);
    }
    check(!draw_polygon_prepare({r.data(),c.data(),4},{{0,0},{2,2}},plan,0));
    std::fesetround(FE_DOWNWARD);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    check(!draw_polygon_prepare({r.data(),c.data(),4},{{0,0},{2,2}},plan,workers));
    check(std::fegetround()==FE_DOWNWARD&&std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO&&plan.labels==labels&&plan.row_offsets==offsets);
    check(std::fesetenv(&prior)==0);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    const auto mxcsr=_mm_getcsr();_mm_setcsr((mxcsr&~0x6000u)|0x2000u);
    const bool accepted=draw_polygon_prepare({r.data(),c.data(),4},{{0,0},{2,2}},plan,workers);
    const auto after_mxcsr=_mm_getcsr();_mm_setcsr(mxcsr);
    check(!accepted&&after_mxcsr==((mxcsr&~0x6000u)|0x2000u));
    check(plan.labels==labels&&plan.row_offsets==offsets&&plan.count==count&&plan.first==first&&plan.shape==shape);
#endif
    check(get_num_threads()==global&&std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
}

// Exact radius2 words from frozen circle-aa/radius-2, pinned0.26 originals.
void circle_aa(int workers){
    const auto global=get_num_threads();
    constexpr std::array<std::intptr_t,24> rr{2,0,2,0,-2,0,-2,0,2,1,1,1,2,1,1,1,-2,-1,-1,-1,-2,-1,-1,-1};
    constexpr std::array<std::intptr_t,24> cc{0,2,0,-2,0,2,0,-2,1,1,2,1,-1,-1,-2,-1,1,1,2,1,-1,-1,-2,-1};
    constexpr std::uint64_t unit=4607182418800017408ULL, complement=4604768947033643348ULL, distance=4598498563450654040ULL;
    for(bool clipped:{false,true}){
        CirclePerimeterAA prepared;
        check(prepare_circle_perimeter_aa({0,0,2,clipped,2,2},prepared));
        std::vector<std::size_t> expected;
        for(std::size_t i=0;i<rr.size();++i)if(!clipped||(rr[i]>=0&&rr[i]<2&&cc[i]>=0&&cc[i]<2))expected.push_back(i);
        const auto count=circle_perimeter_aa_count(prepared,workers);
        check(count==expected.size()&&count<=prepared.max_points);
        for(int step:{1,-2}){
            std::array<std::intptr_t,49> rows,columns;rows.fill(999);columns.fill(999);
            std::array<double,49> weights;weights.fill(-99.25);
            const int start=step>0?1:47;
            circle_perimeter_aa_fill(prepared,{rows.data()+start,columns.data()+start,step,step},weights.data()+start,step,workers);
            for(int i=0;i<49;++i){
                const int delta=i-start;const bool written=delta%step==0&&delta/step>=0&&std::size_t(delta/step)<count;
                if(written){
                    const auto index=expected[std::size_t(delta/step)];std::uint64_t word;std::memcpy(&word,&weights[i],sizeof(word));
                    check(rows[i]==rr[index]&&columns[i]==cc[index]&&word==(index<8?unit:(index%2?distance:complement)));
                }else check(rows[i]==999&&columns[i]==999&&weights[i]==-99.25);
            }
        }
    }
    for(std::intptr_t radius:{-3,0}){
        CirclePerimeterAA prepared;check(prepare_circle_perimeter_aa({0,0,radius},prepared));
        check(circle_perimeter_aa_count(prepared,workers)==8);
        std::array<std::intptr_t,8> rows,columns;std::array<double,8> weights;
        circle_perimeter_aa_fill(prepared,{rows.data(),columns.data(),1,1},weights.data(),1,workers);
        const std::array<std::intptr_t,8> r{radius,0,radius,0,-radius,0,-radius,0},c{0,radius,0,-radius,0,radius,0,-radius};
        check(rows==r&&columns==c&&std::all_of(weights.begin(),weights.end(),[](double v){return v==1.;}));
    }
    CirclePerimeterAA empty;check(prepare_circle_perimeter_aa({0,0,2,true,0,2},empty));
    check(circle_perimeter_aa_count(empty,workers)==0);
    circle_perimeter_aa_fill(empty,{nullptr,nullptr,1,1},nullptr,1,workers);
    const auto prior=empty;constexpr auto high=std::numeric_limits<std::intptr_t>::max(),low=std::numeric_limits<std::intptr_t>::min();
    for(auto q:{CirclePerimeterAARequest{high,0,1},{low,0,1},{0,0,high},{0,0,low}}){
        check(!prepare_circle_perimeter_aa(q,empty));
        check(empty.request.row==prior.request.row&&empty.request.column==prior.request.column&&empty.request.radius==prior.request.radius&&empty.request.clipped==prior.request.clipped&&empty.request.height==prior.request.height&&empty.request.width==prior.request.width&&empty.max_points==prior.max_points&&empty.squared_radius==prior.squared_radius);
    }
    check(get_num_threads()==global);
}
// Frozen half-tie and point words from pinned0.26 original observations.
void line_aa(int workers){
    const auto global=get_num_threads();
    DrawLineAAGeometry geometry;
    check(draw_line_aa_prepare(0,0,2,4,geometry));
    constexpr std::array<std::intptr_t,11> expected_r{0,1,0,1,0,1,2,1,2,1,2};
    constexpr std::array<std::intptr_t,11> expected_c{0,0,1,1,2,2,2,3,3,4,4};
    constexpr std::array<std::uint64_t,11> expected_w{4607182418800017408ULL,4592271743275536696ULL,4603154276835920148ULL,4603154276835920148ULL,4592271743275536696ULL,4607182418800017408ULL,4592271743275536696ULL,4603154276835920148ULL,4603154276835920148ULL,4592271743275536696ULL,4607182418800017408ULL};
    check(draw_line_aa_count(geometry,workers)==expected_r.size());
    for(int step:{1,-2}){
        std::array<std::intptr_t,23> rows,columns;rows.fill(999);columns.fill(999);
        std::array<double,23> weights;weights.fill(-99.25);
        const int start=step>0?1:21;
        draw_line_aa_into(geometry,{rows.data()+start,columns.data()+start,step,step},weights.data()+start,step,workers);
        for(int i=0;i<23;++i){
            const int delta=i-start;const bool written=delta%step==0&&delta/step>=0&&delta/step<11;
            if(written){const auto index=std::size_t(delta/step);std::uint64_t word;std::memcpy(&word,&weights[i],sizeof(word));check(rows[i]==expected_r[index]&&columns[i]==expected_c[index]&&word==expected_w[index]);}
            else check(rows[i]==999&&columns[i]==999&&weights[i]==-99.25);
        }
    }
    check(draw_line_aa_prepare(3,4,3,4,geometry)&&draw_line_aa_count(geometry,workers)==1);
    std::intptr_t row=0,column=0;double weight=0;
    draw_line_aa_into(geometry,{&row,&column,1,1},&weight,1,workers);
    check(row==3&&column==4&&weight==1.0);
    const auto prior=geometry;
    for(const auto endpoints:{std::array<std::intptr_t,4>{std::numeric_limits<std::int32_t>::min(),0,0,0},{0,0,std::numeric_limits<std::int32_t>::max(),0},{0,0,46341,0},{0,0,32768,32768}}){
        check(!draw_line_aa_prepare(endpoints[0],endpoints[1],endpoints[2],endpoints[3],geometry));
        check(geometry.r0==prior.r0&&geometry.c0==prior.c0&&geometry.r1==prior.r1&&geometry.c1==prior.c1&&geometry.dr==prior.dr&&geometry.dc==prior.dc&&geometry.sign_r==prior.sign_r&&geometry.sign_c==prior.sign_c&&geometry.distance==prior.distance);
    }
    check(get_num_threads()==global);
}
void stages(int workers){
    const auto global=get_num_threads(),rounding=std::fegetround(),flags=std::fetestexcept(FE_ALL_EXCEPT);
    std::array<std::intptr_t,7> line_r,line_c;line_r.fill(999);line_c.fill(999);
    draw_line_into(0,0,2,4,{line_r.data()+1,line_c.data()+1,1,1},workers);
    check((line_r==std::array<std::intptr_t,7>{999,0,1,1,2,2,999}));
    check((line_c==std::array<std::intptr_t,7>{999,0,1,2,3,4,999}));
    draw_line_into(2,4,0,0,{line_r.data()+5,line_c.data()+1,-1,1},workers);
    check((line_r==std::array<std::intptr_t,7>{999,0,0,1,1,2,999}));
    check((line_c==std::array<std::intptr_t,7>{999,4,3,2,1,0,999}));
    std::array<std::intptr_t,3> point_r{999,999,999},point_c=point_r;
    draw_line_into(-3,7,-3,7,{point_r.data()+1,point_c.data()+1,1,1},workers);
    check(point_r[0]==999&&point_r[1]==-3&&point_r[2]==999&&point_c[0]==999&&point_c[1]==7&&point_c[2]==999);

    const std::intptr_t starts[]{-1,2,4};const std::size_t lengths[]{2,3,2};
    std::array<std::array<std::intptr_t,14>,3> grids;for(auto& grid:grids)grid.fill(999);
    std::intptr_t* outputs[]{grids[0].data()+1,grids[1].data()+1,grids[2].data()+1};
    draw_rectangle_into(starts,lengths,3,12,outputs,workers);
    for(std::intptr_t c=0;c<3;++c)for(std::intptr_t r=0;r<2;++r)for(std::intptr_t z=0;z<2;++z){
        const auto index=std::size_t((c*2+r)*2+z);check(outputs[0][index]==r-1&&outputs[1][index]==c+2&&outputs[2][index]==z+4);
    }
    for(const auto& grid:grids)check(grid.front()==999&&grid.back()==999);
    const std::size_t empty[]{0,3};
    draw_rectangle_into(starts,empty,2,0,outputs,workers);
    draw_rectangle_into(nullptr,nullptr,0,0,nullptr,workers);
    for(const auto& grid:grids)check(grid.front()==999&&grid.back()==999);
    // Above the existing partition threshold: actual independent SDK blocks.
    const std::size_t big[]{513,517};std::vector<std::intptr_t> r(big[0]*big[1]),c(r.size());std::intptr_t* big_outputs[]{r.data(),c.data()};
    draw_rectangle_into(starts,big,2,r.size(),big_outputs,workers);
    for(std::size_t col=0;col<big[1];++col)for(std::size_t row=0;row<big[0];++row){const auto i=col*big[0]+row;check(r[i]==std::intptr_t(row)-1&&c[i]==std::intptr_t(col)+2);}

    DrawEllipseGrid ellipse{7,11,3.,5.,2.,4.,0.,1.,-7,9};
    std::array<std::uint8_t,79> mask;mask.fill(0x5a);std::array<std::size_t,10> prefix;prefix.fill(999);
    check(draw_ellipse_mask_into(ellipse,mask.data()+1,prefix.data()+1,workers));
    check(mask.front()==0x5a&&mask.back()==0x5a&&prefix.front()==999&&prefix.back()==999&&prefix[1]==0);
    std::vector<std::intptr_t> expected;
    for(std::intptr_t row=0;row<7;++row)for(std::intptr_t col=0;col<11;++col)if(4*(row-3)*(row-3)+(col-5)*(col-5)<16){expected.push_back(row-7);expected.push_back(col+9);}
    check(prefix[8]*2==expected.size());std::vector<std::intptr_t> pairs(expected.size()+2,999);
    draw_ellipse_emit_into(ellipse,mask.data()+1,prefix.data()+1,{pairs.data()+1,pairs.data()+2,2,2},workers);
    check(pairs.front()==999&&pairs.back()==999&&std::equal(expected.begin(),expected.end(),pairs.begin()+1));
    check(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
    for(bool zero_rows:{false,true}){
        auto zero=ellipse;zero.rows=zero_rows?0:5;zero.columns=zero_rows?5:0;std::array<std::size_t,6> empty_prefix;empty_prefix.fill(999);
        check(draw_ellipse_mask_into(zero,nullptr,empty_prefix.data(),workers));
        check(std::all_of(empty_prefix.begin(),empty_prefix.begin()+std::ptrdiff_t(zero.rows+1),[](auto value){return value==0;}));
        draw_ellipse_emit_into(zero,nullptr,empty_prefix.data(),{nullptr,nullptr,2,2},workers);
    }
    std::fesetround(FE_DOWNWARD);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    mask.fill(0x5a);prefix.fill(999);
    check(!draw_ellipse_mask_into(ellipse,mask.data()+1,prefix.data()+1,workers));
    check(std::fegetround()==FE_DOWNWARD&&std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    check(std::all_of(mask.begin(),mask.end(),[](auto value){return value==0x5a;}));
    check(std::all_of(prefix.begin(),prefix.end(),[](auto value){return value==999;}));
    std::fesetround(rounding);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(flags);
    check(get_num_threads()==global&&std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
}
}
int bezier_curve_consumer_controls(int argc,char** argv);
int set_color_consumer_controls();
int main(int argc,char** argv){
    check(argc==1||(argc==2&&(std::string(argv[1])=="--line-aa-only"||std::string(argv[1])=="--circle-aa-only"||std::string(argv[1])=="--polygon-only"||std::string(argv[1])=="--draw-batch-only")));
    const bool line_aa_only=argc==2&&std::string(argv[1])=="--line-aa-only",circle_aa_only=argc==2&&std::string(argv[1])=="--circle-aa-only",polygon_only=argc==2&&std::string(argv[1])=="--polygon-only";
    check(get_num_threads()==1);
    const bool draw_batch_only=argc==2&&std::string(argv[1])=="--draw-batch-only";
    if(!draw_batch_only)for(int workers:{1,4}){set_num_threads(workers);if(!polygon_only){if(!line_aa_only&&!circle_aa_only)stages(workers);if(!circle_aa_only)line_aa(workers);if(!line_aa_only)circle_aa(workers);}if(argc==1||polygon_only)polygon(workers);}
    set_num_threads(1);check(get_num_threads()==1);
    if(argc==1||draw_batch_only){
        char program[]="drawing SDK",fixture[]=SKIMAGEXX_BEZIER_FIXTURE_PATH,dispatch[]="auto";
        char* arguments[]{program,fixture,dispatch};
        check(bezier_curve_consumer_controls(3,arguments)==0);
        check(set_color_consumer_controls()==0);
        check(get_num_threads()==1);
    }
    std::cout<<"installed drawing SDK controls="<<checks<<"; workers=1,4; default=1; AVX2="<<has_avx2()<<"; line_aa_only="<<line_aa_only<<"; circle_aa_only="<<circle_aa_only<<"; polygon_only="<<polygon_only<<"\n";
}
