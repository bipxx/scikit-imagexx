#include "block_reduce_internal.hpp"
#include "half.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
namespace {
std::size_t checks=0;std::ofstream journal;
void require(bool value){++checks;if(!value)throw std::runtime_error("block sum control "+std::to_string(checks));}
template<class T>void save(const std::vector<T>& values,unsigned status){
    journal.write(reinterpret_cast<const char*>(&status),sizeof(status));
    if(!values.empty())journal.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(T)));
}
template<class T>void ordinary(PixelType type,std::size_t rows,std::size_t columns,std::size_t fy,std::size_t fx,bool fortran=false){
    const auto nr=rows/fy,nc=columns/fx;std::vector<T> input(rows*columns,T(.25)),output(nr*nc+2,T(73));
    const auto before=input;detail::BlockReduceInput view{input.data(),type,false,{rows,columns}};
    const std::vector<std::size_t> strides=fortran ? std::vector<std::size_t>{sizeof(T),nr*sizeof(T)}:std::vector<std::size_t>{nc*sizeof(T),sizeof(T)};
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const int prior=std::fetestexcept(FE_ALL_EXCEPT);
    const auto status=detail::block_reduce_sum(view,{fy,fx},output.data()+1,strides);
    require(std::fetestexcept(FE_ALL_EXCEPT)==prior);std::feclearexcept(FE_ALL_EXCEPT);
    require(status==0&&input==before);require(output.front()==T(73)&&output.back()==T(73));
    for(std::size_t i=1;i+1<output.size();++i)require(output[i]==T(fy*fx)*T(.25));
    save(output,status);
}
template<class T>void integer(PixelType type){
    std::array<T,8> input{};for(std::size_t i=0;i<8;++i)input[i]=T(i+1);
    std::vector<std::uint64_t> output(4,91);
    const auto status=detail::block_reduce_sum({input.data(),type,false,{2,4}},{2,2},output.data()+1,{16,8});
    require(status==0&&output==std::vector<std::uint64_t>({91,14,22,91}));save(output,status);
}
void special_integer(){
    const std::array<std::int64_t,4> signed_values{INT64_MAX,INT64_MAX,INT64_MIN,-3};
    const std::array<std::uint64_t,4> unsigned_values{UINT64_MAX,UINT64_MAX,3,7};
    const std::array<std::uint8_t,4> booleans{0,2,128,255};std::vector<std::uint64_t> result(1);
    require(detail::block_reduce_sum({signed_values.data(),PixelType::int64,false,{4}},{4},result.data(),{8})==0);
    require(result[0]==std::uint64_t(INT64_MAX)-4);save(result,0);
    require(detail::block_reduce_sum({unsigned_values.data(),PixelType::uint64,false,{4}},{4},result.data(),{8})==0);
    require(result[0]==8);save(result,0);
    require(detail::block_reduce_sum({booleans.data(),PixelType::boolean,false,{4}},{4},result.data(),{8})==0);
    require(result[0]==3);save(result,0);
    const std::array<std::int8_t,4> negative{-1,-2,-3,-4};
    require(detail::block_reduce_sum({negative.data(),PixelType::int8,false,{4}},{4},result.data(),{8})==0);
    require(result[0]==std::uint64_t(-10));save(result,0);
}
void half_and_rank(){
    std::array<std::uint16_t,8> half{};half.fill(0x3400);std::vector<std::uint16_t> output(2);
    require(detail::block_reduce_sum({half.data(),PixelType::uint16,true,{2,4}},{1,4},output.data(),{2,2})==0);
    require(output==std::vector<std::uint16_t>({0x3c00,0x3c00}));save(output,0);
    double scalar=-0.;std::vector<double> out(1,72);
    require(detail::block_reduce_sum({&scalar,PixelType::float64,false,{}},{},out.data(),{})==0);
    require(out[0]==0&&!std::signbit(out[0]));save(out,0);
    std::vector<float> volume(2*4*6*8,.25f),result(2*2*2*2);
    require(detail::block_reduce_sum({volume.data(),PixelType::float32,false,{2,4,6,8}},{1,2,3,4},result.data(),{32,16,8,4})==0);
    for(auto value:result)require(value==6);save(result,0);
    require(detail::block_reduce_sum({nullptr,PixelType::float32,false,{2,0,4}},{1,2,2},nullptr,{0,0,0})==0);
}

void half_segment_regression(){
    // Frozen original dtype-float16 cell [0,2], with its actual 6x8/2x2 geometry.
    // HALF_add stores a half after each iterator segment, not once per block.
    std::vector<std::uint16_t> input(48,0),output(14,0x5555);
    const std::array<float,4> values{-.71435546875f,-.5712890625f,.428466796875f,.5712890625f};
    const std::array<std::size_t,4> locations{4,5,12,13};
    for(std::size_t i=0;i<4;++i)input[locations[i]]=detail::encode_half(values[i]);
    const auto before=input;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto prior=std::fetestexcept(FE_ALL_EXCEPT);
    const auto status=detail::block_reduce_sum({input.data(),PixelType::uint16,true,{6,8}},{2,2},output.data()+1,{8,2});
    require(std::fetestexcept(FE_ALL_EXCEPT)==prior);std::feclearexcept(FE_ALL_EXCEPT);
    require(status==0&&input==before);require(output.front()==0x5555&&output.back()==0x5555);
    save(output,status);require(output[3]==0xb491);
    // With no intervening output axis, all four inputs form one inner segment.
    std::array<std::uint16_t,4> contiguous{};
    for(std::size_t i=0;i<4;++i)contiguous[i]=detail::encode_half(values[i]);
    std::vector<std::uint16_t> single(1);
    require(detail::block_reduce_sum({contiguous.data(),PixelType::uint16,true,{2,2}},{2,2},single.data(),{2,2})==0);
    require(single[0]==0xb493);save(single,0);
    // Intermediate half overflow is observable even when the exact block sum is zero.
    input[4]=input[5]=0x7bff;input[12]=input[13]=0xfbff;
    const auto overflow=detail::block_reduce_sum({input.data(),PixelType::uint16,true,{6,8}},{2,2},output.data()+1,{8,2});
    require(overflow==2&&output[3]==0x7c00);save(output,overflow);
}
template<class T>void cancellation(PixelType type){
    for(std::size_t n:{7,8,9,127,128,129,8193}){
        std::vector<T> input(n,1),out(1);input.front()=T(16777216);input.back()=-input.front();
        const auto status=detail::block_reduce_sum({input.data(),type,false,{n}},{n},out.data(),{sizeof(T)});
        const long double sum=static_cast<long double>(n-2),S=33554432.L+sum;
        const long double u=std::numeric_limits<T>::epsilon()/2.L,g=n*u/(1-n*u);
        require(status==0&&std::abs(static_cast<long double>(out[0])-sum)<=2*g*S);save(out,status);
    }
}
template<class T>void exceptions(PixelType type){
    for(int kind:{0,1}){
        std::vector<T> input(512*1024,T(0)),out(256*512);
        input[input.size()-1]=kind ? std::numeric_limits<T>::max():-std::numeric_limits<T>::infinity();
        input[input.size()-2]=kind ? std::numeric_limits<T>::max():std::numeric_limits<T>::infinity();
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto before=std::fetestexcept(FE_ALL_EXCEPT);
        const auto status=detail::block_reduce_sum({input.data(),type,false,{512,1024}},{2,2},out.data(),{512*sizeof(T),sizeof(T)});
        require(std::fetestexcept(FE_ALL_EXCEPT)==before);std::feclearexcept(FE_ALL_EXCEPT);
        require(status==(kind?2u:8u));require(kind?std::isinf(out.back()):std::isnan(out.back()));save(out,status);
    }
    std::vector<T> input(4,T(-0.)),out(1);
    require(detail::block_reduce_sum({input.data(),type,false,{4}},{4},out.data(),{sizeof(T)})==0);
    require(out[0]==0&&!std::signbit(out[0]));save(out,0);
}
template<class T>void packed_cancellation(PixelType type){
    for(std::size_t fx:{1,3,43}){
        const std::size_t columns=19*fx,volume=3*fx;std::vector<T> input(6*columns),output(38);
        for(std::size_t row=0;row<2;++row)for(std::size_t col=0;col<19;++col){
            for(std::size_t j=0;j<volume;++j){
                T value=T(int((j+col)%7)-3)*T(.125);
                if(!j)value=std::is_same_v<T,float>?T(16777216):T(9007199254740992.);
                if(j+1==volume)value=-(std::is_same_v<T,float>?T(16777216):T(9007199254740992.));
                input[(row*3+j/fx)*columns+col*fx+j%fx]=value;
            }
        }
        const auto status=detail::block_reduce_sum({input.data(),type,false,{6,columns}},{3,fx},output.data(),{19*sizeof(T),sizeof(T)});
        require(status==0);
        for(std::size_t row=0;row<2;++row)for(std::size_t col=0;col<19;++col){
            long double exact=0,magnitude=0;
            for(std::size_t j=0;j<volume;++j){const auto value=input[(row*3+j/fx)*columns+col*fx+j%fx];exact+=value;magnitude+=std::abs(static_cast<long double>(value));}
            const long double u=std::numeric_limits<T>::epsilon()/2.L,g=volume*u/(1-volume*u);
            require(std::abs(static_cast<long double>(output[row*19+col])-exact)<=2*g*magnitude);
        }
        save(output,status);
    }
}
template<class T>void packed_zero_sign(PixelType type){
    for(auto shape:{std::array<std::size_t,4>{6,8,2,2},{6,18,2,2},{3,27,1,3},{512,1024,2,2}}){
        const auto rows=shape[0],columns=shape[1],fy=shape[2],fx=shape[3];
        const auto nr=rows/fy,nc=columns/fx;
        std::vector<T> input(rows*columns,-T(0)),output(nr*nc+2,T(73));
        const auto status=detail::block_reduce_sum({input.data(),type,false,{rows,columns}},
            {fy,fx},output.data()+1,{nc*sizeof(T),sizeof(T)});
        require(status==0&&output.front()==T(73)&&output.back()==T(73));
        for(std::size_t i=1;i+1<output.size();++i)require(output[i]==T(0)&&!std::signbit(output[i]));
        save(output,status);
    }
}
void invalid_geometry(){
    const std::array<float,4> input{1,2,3,4};std::array<float,4> out{91,91,91,91};
    for(int kind:{0,1,2,3}){
        bool rejected=false;try{
            detail::BlockReduceInput in{input.data(),PixelType::float32,false,{2,2}};
            detail::block_reduce_sum(in,kind==0?std::vector<std::size_t>{0,2}:kind==1?std::vector<std::size_t>{3,2}:std::vector<std::size_t>{1,1},
                                    kind==3?nullptr:out.data(),kind==2?std::vector<std::size_t>{4,4}:std::vector<std::size_t>{8,4});
        }catch(const std::invalid_argument&){rejected=true;}
        require(rejected&&out==std::array<float,4>{91,91,91,91});
    }
}
}
int main(int argc,char**argv){
    if(argc!=2)return 2;journal.open(argv[1],std::ios::binary);require(bool(journal));
    require(get_num_threads()==1);
    std::vector<int> workers{1};
#ifdef _OPENMP
    workers.push_back(4);
#else
    bool refused=false;
    try{set_num_threads(4);}catch(const std::invalid_argument& error){
        refused=std::string(error.what())=="this build has no OpenMP support";
    }
    require(refused && get_num_threads()==1);
#endif
    for(int worker:workers){set_num_threads(worker);
        integer<std::int8_t>(PixelType::int8);integer<std::uint8_t>(PixelType::uint8);
        integer<std::int16_t>(PixelType::int16);integer<std::uint16_t>(PixelType::uint16);
        integer<std::int32_t>(PixelType::int32);integer<std::uint32_t>(PixelType::uint32);
        integer<std::int64_t>(PixelType::int64);integer<std::uint64_t>(PixelType::uint64);
        special_integer();half_and_rank();invalid_geometry();cancellation<float>(PixelType::float32);cancellation<double>(PixelType::float64);
        for(auto shape: {std::array<std::size_t,4>{4,96,2,3}, {6,688,3,43}, {4,96,1,1}, {512,1024,2,2}}){
            ordinary<float>(PixelType::float32,shape[0],shape[1],shape[2],shape[3]);
            ordinary<double>(PixelType::float64,shape[0],shape[1],shape[2],shape[3]);
        }
        ordinary<float>(PixelType::float32,4,96,2,3,true);ordinary<double>(PixelType::float64,4,96,2,3,true);
        packed_cancellation<float>(PixelType::float32);packed_cancellation<double>(PixelType::float64);
        exceptions<float>(PixelType::float32);exceptions<double>(PixelType::float64);
        half_segment_regression();
    }
    for(int worker:workers){set_num_threads(worker);packed_zero_sign<float>(PixelType::float32);packed_zero_sign<double>(PixelType::float64);}
    set_num_threads(1);journal.close();std::cout<<checks<<" block sum controls PASS avx2="<<has_avx2()<<" workers="<<get_num_threads()<<'\n';
}
