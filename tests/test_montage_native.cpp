#include <skimagexx/montage.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0,cases=0;
void check(bool value){++checks;if(!value)throw std::runtime_error("montage native check "+std::to_string(checks));}
std::uint64_t word(std::ifstream& f){std::uint64_t value=0;f.read(reinterpret_cast<char*>(&value),8);check(bool(f));return value;}
std::vector<unsigned char> bytes(std::ifstream& f,std::size_t n){std::vector<unsigned char> v(n);if(n)f.read(reinterpret_cast<char*>(v.data()),n);check(bool(f));return v;}
void exercise(const std::array<std::size_t,4>& shape,const std::array<std::ptrdiff_t,4>& strides,
              std::size_t itemsize,std::size_t origin,const std::vector<unsigned char>& backing,
              MontageGrid grid,std::size_t out_rows,std::size_t out_columns,
              const std::vector<unsigned char>& fill,int workers,std::ofstream& stream,
              const std::vector<unsigned char>* genuine=nullptr,std::size_t output_offset=16){
    ++cases;const auto K=shape[0],H=shape[1],W=shape[2],C=shape[3];
    std::vector<unsigned char> expected(out_rows*out_columns*C*itemsize);
    // Output-centric independent oracle; kernel iterates source image rows.
    for(std::size_t y=0;y<out_rows;++y)for(std::size_t x=0;x<out_columns;++x){
        bool selected=false;std::size_t image=0,row=0,column=0;
        if(y>=grid.padding&&x>=grid.padding){
            const auto ty=(y-grid.padding)/(H+grid.padding),tx=(x-grid.padding)/(W+grid.padding);
            row=(y-grid.padding)%(H+grid.padding);column=(x-grid.padding)%(W+grid.padding);
            image=ty*grid.columns+tx;
            selected=ty<grid.rows&&tx<grid.columns&&row<H&&column<W&&image<K;
        }
        for(std::size_t channel=0;channel<C;++channel){
            const unsigned char* from;
            if(selected){
                const auto offset=std::ptrdiff_t(origin)+std::ptrdiff_t(image)*strides[0]+std::ptrdiff_t(row)*strides[1]+std::ptrdiff_t(column)*strides[2]+std::ptrdiff_t(channel)*strides[3];
                from=backing.data()+offset;
            }else from=fill.data()+channel*itemsize;
            std::memcpy(expected.data()+((y*out_columns+x)*C+channel)*itemsize,from,itemsize);
        }
    }
    if(genuine)check(expected==*genuine);
    const auto before_input=backing,before_fill=fill;const auto global=get_num_threads();
    MontageInput input{backing.empty()?nullptr:backing.data()+origin,{K,H,W,C},{strides[0],strides[1],strides[2],strides[3]},itemsize};
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        check(std::fesetround(rounding)==0);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID|FE_DIVBYZERO);
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        std::vector<unsigned char> actual(expected.size()+output_offset+16,0xa5);
        montage_into(input,grid,fill.empty()?nullptr:fill.data(),{expected.empty()?nullptr:actual.data()+output_offset,out_rows,out_columns},workers);
        check(expected.empty()||std::memcmp(actual.data()+output_offset,expected.data(),expected.size())==0);
        check(std::all_of(actual.begin(),actual.begin()+output_offset,[](auto value){return value==0xa5;}));
        check(std::all_of(actual.end()-16,actual.end(),[](auto value){return value==0xa5;}));
        check(backing==before_input&&fill==before_fill);
        check(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags&&get_num_threads()==global);
        std::uint64_t hash=1469598103934665603ULL;
        for(std::size_t i=0;i<expected.size();++i){hash^=actual[i+output_offset];hash*=1099511628211ULL;}
        stream.write(reinterpret_cast<const char*>(&hash),8);
    }
    std::fesetround(FE_TONEAREST);std::feclearexcept(FE_ALL_EXCEPT);
}
void control(std::array<std::size_t,4> shape,std::size_t itemsize,int layout,MontageGrid grid,bool zero,int workers,std::ofstream& stream,bool unaligned=false){
    std::array<std::ptrdiff_t,4> strides{};std::size_t step=itemsize;
    if(layout==1)for(std::size_t axis=0;axis<4;++axis){strides[axis]=std::ptrdiff_t(step);step*=shape[axis];}
    else for(std::size_t axis=4;axis-->0;){if(layout==2)step*=2;strides[axis]=std::ptrdiff_t(step);step*=shape[axis];}
    if(layout==3)for(auto& stride:strides)stride=-stride;
    if(layout==4)strides[3]=0;
    if(layout==5)strides[0]=strides[1]=strides[3]=0;
    const bool empty=std::any_of(shape.begin(),shape.end(),[](auto value){return value==0;});
    std::ptrdiff_t low=0,high=0;
    if(!empty)for(std::size_t axis=0;axis<4;++axis){const auto extent=std::ptrdiff_t(shape[axis]-1)*strides[axis];low+=std::min(std::ptrdiff_t(0),extent);high+=std::max(std::ptrdiff_t(0),extent);}
    const auto origin=std::size_t(-low)+(unaligned?1:0);std::vector<unsigned char> backing(empty?0:std::size_t(high-low)+itemsize+(unaligned?1:0),0xda);
    for(std::size_t k=0;k<shape[0];++k)for(std::size_t y=0;y<shape[1];++y)for(std::size_t x=0;x<shape[2];++x)for(std::size_t c=0;c<shape[3];++c){
        const std::size_t values[]={k,y,x,c};std::size_t code=17;
        for(std::size_t axis=0;axis<4;++axis)code=code*37+(strides[axis]?values[axis]:0);
        const auto offset=std::ptrdiff_t(origin)+std::ptrdiff_t(k)*strides[0]+std::ptrdiff_t(y)*strides[1]+std::ptrdiff_t(x)*strides[2]+std::ptrdiff_t(c)*strides[3];
        for(std::size_t b=0;b<itemsize;++b)backing[std::size_t(offset)+b]=static_cast<unsigned char>(code+b*11);
    }
    std::vector<unsigned char> fill(shape[3]*itemsize);
    if(!zero)for(std::size_t i=0;i<fill.size();++i)fill[i]=static_cast<unsigned char>(43+i*13);
    exercise(shape,strides,itemsize,origin,backing,grid,(shape[1]+grid.padding)*grid.rows+grid.padding,(shape[2]+grid.padding)*grid.columns+grid.padding,fill,workers,stream,nullptr,unaligned?17:16);
}
void genuine(const char* path,int workers,std::ofstream& stream){
    std::ifstream f(path,std::ios::binary);check(bool(f));const auto count=word(f);
    for(std::size_t record=0;record<count;++record){
        std::array<std::size_t,4> shape{};std::array<std::ptrdiff_t,4> strides{};
        for(auto& n:shape)n=word(f);
        for(auto& n:strides){const auto raw=word(f);std::int64_t signed_value;std::memcpy(&signed_value,&raw,8);n=std::ptrdiff_t(signed_value);}
        const auto itemsize=word(f);const MontageGrid grid{word(f),word(f),word(f)};
        const auto rows=word(f),columns=word(f),origin=word(f),input_bytes=word(f),fill_bytes=word(f),expected_bytes=word(f);
        auto backing=bytes(f,input_bytes),fill=bytes(f,fill_bytes),expected=bytes(f,expected_bytes);
        exercise(shape,strides,itemsize,origin,backing,grid,rows,columns,fill,workers,stream,&expected);
    }
    check(f.peek()==std::char_traits<char>::eof());
}
}
int main(int argc,char** argv){
    if(argc!=2&&argc!=3)throw std::runtime_error("usage: montage-native output [frozen-fixtures]");
    std::ofstream stream(argv[1],std::ios::binary);check(bool(stream));
    for(int workers:{1,4}){
        if(argc==3)genuine(argv[2],workers,stream);
        for(std::size_t width:{1,2,4,8,16}){
            for(int layout=0;layout<6;++layout)control({5,3,7,3},width,layout,{2,3,1},layout%2==0,workers,stream);
            control({3,2,5,1},width,4,{2,2,0},false,workers,stream);
            control({0,2,3,1},width,0,{0,0,1},false,workers,stream);
            control({3,0,4,2},width,0,{2,2,1},true,workers,stream);
            control({3,2,0,2},width,0,{2,2,1},false,workers,stream);
            control({3,2,4,0},width,0,{2,2,1},false,workers,stream);
            control({0,2,3,1},width,0,{0,0,0},false,workers,stream);
        }
        control({9,64,96,3},4,0,{3,3,1},false,workers,stream);
    }
    // Append controls after the complete old stream so its words stay intact.
    for(int workers:{1,4}) {
        for(std::size_t width:{1,2,4,8,16}) {
            control({2,3,7,1},width,2,{1,2,1},false,workers,stream,true);
            control({2,3,7,3},width,3,{1,2,1},false,workers,stream,true);
            control({2,3,7,3},width,1,{1,2,1},false,workers,stream,true);
        }
        control({2,3,7,3},4,2,{1,2,1},false,workers,stream,true);
    }
    check(bool(stream));std::cout<<"montage native cases="<<cases<<" checks="<<checks<<" PASS\n";
}
