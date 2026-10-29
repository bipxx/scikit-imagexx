#include "skimagexx/draw_coordinates.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0,fixture_count=0;std::uint64_t words_hash=UINT64_C(1469598103934665603);
std::string context;
void require(bool value){++checks;if(!value)throw std::runtime_error(context+" control="+std::to_string(checks));}
void hash_word(std::intptr_t word){std::uint64_t bits;std::memcpy(&bits,&word,8);for(unsigned k=0;k<8;++k){words_hash^=(bits>>(k*8))&255;words_hash*=UINT64_C(1099511628211);}}
template<class T>T read(std::ifstream& in){T result;in.read(reinterpret_cast<char*>(&result),sizeof(result));require(bool(in));return result;}
std::string label(std::ifstream& in){const auto n=read<std::uint64_t>(in);std::string result(n,' ');in.read(result.data(),std::streamsize(n));require(bool(in));return result;}
std::vector<std::intptr_t> expected(std::ifstream& in,std::size_t n){std::vector<std::intptr_t> result(n);in.read(reinterpret_cast<char*>(result.data()),std::streamsize(n*8));require(bool(in));return result;}
void ellipse_gold(const DrawEllipseGrid& grid,const std::vector<std::intptr_t>& rr,const std::vector<std::intptr_t>& cc,int workers){
    const auto n=grid.rows*grid.columns;std::vector<std::uint8_t> mask(n+2,0x5a);std::vector<std::size_t> prefix(grid.rows+3,999999);
    const auto old_round=std::fegetround(),old_flags=std::fetestexcept(FE_ALL_EXCEPT);
    require(draw_ellipse_mask_into(grid,mask.data()+1,prefix.data()+1,workers));
    require(std::fegetround()==old_round&&std::fetestexcept(FE_ALL_EXCEPT)==old_flags);
    require(mask.front()==0x5a&&mask.back()==0x5a&&prefix.front()==999999&&prefix.back()==999999);
    require(prefix[1]==0&&prefix[grid.rows+1]==rr.size()&&rr.size()==cc.size());
    std::vector<std::intptr_t> pairs(rr.size()*2+2,777777);
    draw_ellipse_emit_into(grid,mask.data()+1,prefix.data()+1,{pairs.data()+1,pairs.data()+2,2,2},workers);
    require(pairs.front()==777777&&pairs.back()==777777);
    for(std::size_t i=0;i<rr.size();++i){require(pairs[1+i*2]==rr[i]&&pairs[2+i*2]==cc[i]);hash_word(pairs[1+i*2]);hash_word(pairs[2+i*2]);}
}
void saved(const char* path,int workers){
    std::ifstream in(path,std::ios::binary);require(bool(in));const auto records=read<std::uint64_t>(in);
    require(records==68);
    for(std::size_t record=0;record<records;++record){context=label(in);const auto kind=read<std::uint64_t>(in);++fixture_count;
        if(kind==0){const auto r0=read<std::int64_t>(in),c0=read<std::int64_t>(in),r1=read<std::int64_t>(in),c1=read<std::int64_t>(in);const auto n=read<std::uint64_t>(in);
            const auto rr=expected(in,n),cc=expected(in,n);std::vector<std::intptr_t> r(n+2,777777),c(n+2,777777);
            draw_line_into(r0,c0,r1,c1,{r.data()+1,c.data()+1,1,1},workers);
            require(r.front()==777777&&r.back()==777777&&c.front()==777777&&c.back()==777777);
            for(std::size_t i=0;i<n;++i){require(r[i+1]==rr[i]&&c[i+1]==cc[i]);hash_word(r[i+1]);hash_word(c[i+1]);}
        }else if(kind==1){const auto ndim=read<std::uint64_t>(in),n=read<std::uint64_t>(in);std::vector<std::intptr_t> starts(ndim);std::vector<std::size_t> lengths(ndim);
            for(auto& v:starts)v=read<std::int64_t>(in);for(auto& v:lengths)v=read<std::uint64_t>(in);
            std::vector<std::vector<std::intptr_t>> gold,outputs(ndim,std::vector<std::intptr_t>(n+2,777777));std::vector<std::intptr_t*> pointers;
            for(auto& output:outputs)pointers.push_back(output.data()+1);for(std::size_t d=0;d<ndim;++d)gold.push_back(expected(in,n));
            draw_rectangle_into(starts.data(),lengths.data(),ndim,n,pointers.data(),workers);
            for(std::size_t d=0;d<ndim;++d){require(outputs[d].front()==777777&&outputs[d].back()==777777);for(std::size_t i=0;i<n;++i){require(outputs[d][i+1]==gold[d][i]);hash_word(outputs[d][i+1]);}}
        }else{require(kind==2);DrawEllipseGrid grid;grid.rows=read<std::uint64_t>(in);grid.columns=read<std::uint64_t>(in);
            grid.center_row=read<double>(in);grid.center_column=read<double>(in);grid.row_radius=read<double>(in);grid.column_radius=read<double>(in);grid.sin_alpha=read<double>(in);grid.cos_alpha=read<double>(in);grid.row_offset=read<std::int64_t>(in);grid.column_offset=read<std::int64_t>(in);
            const auto n=read<std::uint64_t>(in);const auto rr=expected(in,n),cc=expected(in,n);ellipse_gold(grid,rr,cc,workers);
        }
    }
    require(in.peek()==std::char_traits<char>::eof());
}
void lines(int workers){
    context="independent Bresenham integer-rounding oracle";
    for(std::intptr_t dr=-32;dr<=32;++dr)for(std::intptr_t dc=-32;dc<=32;++dc){
        const auto ar=dr<0?-dr:dr,ac=dc<0?-dc:dc,major=std::max(ar,ac),minor=std::min(ar,ac);const auto n=std::size_t(major+1);
        std::vector<std::intptr_t> rr(3*n+2,777777),cc(3*n+2,777777);
        draw_line_into(-7,3,-7+dr,3+dc,{rr.data()+3*n,cc.data()+1,-3,3},workers);
        for(std::intptr_t k=0;k<=major;++k){const auto m=major?(2*minor*k+major)/(2*major):0;const auto r=-7+(dr<0?-1:1)*(ar>ac?k:m),c=3+(dc<0?-1:1)*(ar>ac?m:k);
            require(rr[3*n-3*std::size_t(k)]==r&&cc[1+3*std::size_t(k)]==c);}
        for(std::size_t i=0;i<rr.size();++i){if(i<3||i%3)require(rr[i]==777777);if(i<1||(i-1)%3||i>=3*n)require(cc[i]==777777);}
    }
}
void rectangles(int workers){
    context="independent NDxy grids";
    for(std::size_t ndim:{0u,1u,2u,3u,4u,8u,17u,32u}){
        std::vector<std::size_t> lengths(ndim,1),shape,pitch(ndim,1);std::vector<std::intptr_t> starts(ndim,-3);if(ndim)lengths[0]=7;if(ndim>1)lengths[1]=5;if(ndim>2)lengths[2]=3;
        for(bool empty:{false,true}){if(empty&&ndim)lengths.back()=0;shape=lengths;if(ndim>1)std::swap(shape[0],shape[1]);std::size_t n=ndim?1:0;for(auto d:lengths)n*=d;
            std::vector<std::vector<std::intptr_t>> output(ndim,std::vector<std::intptr_t>(n+2,777777));std::vector<std::intptr_t*> pointers;for(auto& o:output)pointers.push_back(o.data()+1);
            draw_rectangle_into(starts.data(),lengths.data(),ndim,n,pointers.data(),workers);
            for(std::size_t i=0;i<n;++i){auto remainder=i;std::vector<std::size_t> position(ndim);for(auto d=ndim;d-->0;){position[d]=remainder%shape[d];remainder/=shape[d];}if(ndim>1)std::swap(position[0],position[1]);for(std::size_t d=0;d<ndim;++d)require(output[d][i+1]==starts[d]+std::intptr_t(position[d]));}
            for(auto& o:output)require(o.front()==777777&&o.back()==777777);
        }
    }
    // Independent large blocks exercise actual optional row/block partitioning.
    std::size_t lengths[]{513,517};std::intptr_t starts[]{-19,5};std::vector<std::intptr_t> a(lengths[0]*lengths[1]),b(a.size());std::intptr_t* outputs[]{a.data(),b.data()};
    draw_rectangle_into(starts,lengths,2,a.size(),outputs,workers);
    for(std::size_t j=0;j<lengths[1];++j)for(std::size_t i=0;i<lengths[0];++i)require(a[j*lengths[0]+i]==starts[0]+std::intptr_t(i)&&b[j*lengths[0]+i]==starts[1]+std::intptr_t(j));
}
void ellipses(int workers){
    context="independent dyadic strict ellipse oracle";
    for(std::size_t rows:{0u,1u,3u,8u,17u,513u})for(std::size_t cols:{0u,1u,3u,4u,5u,7u,8u,9u,33u,517u})for(bool rotated:{false,true}){
        DrawEllipseGrid grid{rows,cols,4.,8.,4.,8.,rotated?1.:0.,rotated?0.:1.,-5,3};std::vector<std::intptr_t> rr,cc;
        for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){const auto dr=std::intptr_t(r)-4,dc=std::intptr_t(c)-8;const auto u=rotated?dc:dr,v=rotated?dr:-dc;if(4*u*u+v*v<64){rr.push_back(std::intptr_t(r)-5);cc.push_back(std::intptr_t(c)+3);}}
        ellipse_gold(grid,rr,cc,workers);
    }
    context="FP admission/state/refusal";
    const auto old_round=std::fegetround(),old_flags=std::fetestexcept(FE_ALL_EXCEPT);
    for(int mode:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO})for(int flags:{FE_DIVBYZERO,FE_DIVBYZERO|FE_INVALID}){
        for(std::uint64_t bits:{UINT64_C(0x3ff0000000000000),UINT64_C(0),UINT64_C(0xbff0000000000000),UINT64_C(0x7ff0000000000000),UINT64_C(0x7ff8000000000001),UINT64_C(0x7ff0000000000001)}){
            double radius;std::memcpy(&radius,&bits,8);DrawEllipseGrid grid{3,5,1.,2.,radius,2.,0.,1.,0,0};std::array<std::uint8_t,17> mask;mask.fill(0x5a);std::array<std::size_t,6> prefix;prefix.fill(999999);
            std::fesetround(mode);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(flags);const auto before=std::fetestexcept(FE_ALL_EXCEPT);
            const bool accepted=draw_ellipse_mask_into(grid,mask.data()+1,prefix.data()+1,workers);
            require(accepted==(mode==FE_TONEAREST&&bits==UINT64_C(0x3ff0000000000000)));
            require(std::fegetround()==mode&&std::fetestexcept(FE_ALL_EXCEPT)==before);require(mask.front()==0x5a&&mask.back()==0x5a&&prefix.front()==999999&&prefix.back()==999999);
        }
    }
    std::fesetround(FE_TONEAREST);
    for(double radius:{1e-300,1e300}){
        DrawEllipseGrid grid{3,5,1.,2.,radius,radius,0.,1.,0,0};std::array<std::uint8_t,15> mask{};std::array<std::size_t,4> prefix{};
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);require(!draw_ellipse_mask_into(grid,mask.data(),prefix.data(),workers));require(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    }
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(old_flags);std::fesetround(old_round);
}
void strided_emit_and_ftz(int workers){
    context="signed-stride ellipse emission and FTZ refusal";
    DrawEllipseGrid grid{7,11,3.,5.,2.,4.,0.,1.,-7,9};
    std::vector<std::uint8_t> mask(grid.rows*grid.columns);std::vector<std::size_t> prefix(grid.rows+1);
    require(draw_ellipse_mask_into(grid,mask.data(),prefix.data(),workers));
    std::vector<std::intptr_t> rr,cc;
    for(std::intptr_t r=0;r<7;++r)for(std::intptr_t c=0;c<11;++c)if(4*(r-3)*(r-3)+(c-5)*(c-5)<16){rr.push_back(r-7);cc.push_back(c+9);}
    require(prefix.back()==rr.size());const auto n=rr.size();
    std::vector<std::intptr_t> a(3*n+2,777777),b(3*n+2,777777);
    draw_ellipse_emit_into(grid,mask.data(),prefix.data(),{a.data()+3*n,b.data()+1,-3,3},workers);
    for(std::size_t i=0;i<n;++i)require(a[3*n-3*i]==rr[i]&&b[1+3*i]==cc[i]);
    for(std::size_t i=0;i<a.size();++i){if(i<3||i%3)require(a[i]==777777);if(i<1||(i-1)%3||i>=3*n)require(b[i]==777777);}
#if defined(__SSE2__) || defined(_M_X64)
    const auto original=_mm_getcsr();_mm_setcsr(original|0x8040);
    std::fill(mask.begin(),mask.end(),0x5a);std::fill(prefix.begin(),prefix.end(),999999);
    require(!draw_ellipse_mask_into(grid,mask.data(),prefix.data(),workers));
    require(_mm_getcsr()==(original|0x8040));
    require(std::all_of(mask.begin(),mask.end(),[](auto v){return v==0x5a;}));
    require(std::all_of(prefix.begin(),prefix.end(),[](auto v){return v==999999;}));
    _mm_setcsr(original);
#endif
}

}
int main(int argc,char** argv){
    static_assert(sizeof(std::intptr_t)==8,"frozen coordinate words are intp64");require(get_num_threads()==1);
    for(int workers:{1,4}){
#ifdef SKIMAGEXX_DRAW_PORTABLE
        if(workers==4){bool rejected=false;try{set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}require(rejected);continue;}
#endif
        set_num_threads(workers);for(int i=1;i<argc;++i)saved(argv[i],workers);
    }
    std::cout<<"draw frozen records="<<fixture_count<<"; coordinate word hash="<<words_hash<<'\n';
    for(int workers:{1,4}){
#ifdef SKIMAGEXX_DRAW_PORTABLE
        if(workers==4)continue;
#endif
        set_num_threads(workers);lines(workers);rectangles(workers);ellipses(workers);strided_emit_and_ftz(workers);
    }
    set_num_threads(1);std::cout<<"draw native controls="<<checks<<"; default worker=1; AVX2="<<has_avx2()<<'\n';
}
