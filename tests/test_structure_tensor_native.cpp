#include "skimagexx/structure_tensor.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
void require(bool condition){if(!condition)throw std::runtime_error("native structure tensor check failed");}
template<class T> void impulse(skimagexx::PixelType type,T peak){
    using namespace skimagexx;
    std::vector<T> input(25,T(0));input[12]=peak;
    using Output=std::conditional_t<std::is_same_v<T,float>,float,double>;
    std::array<std::vector<Output>,3> fields;std::array<void*,3> pointers;
    for(std::size_t f=0;f<3;++f){fields[f].resize(25);pointers[f]=fields[f].data();}
    for(bool xy:{false,true}){
        require(structure_tensor_into(input.data(),type,pointers.data(),{5,5},{},{GaussianBorder::constant,GaussianBorder::constant},0,xy));
        const auto rr=xy ? 2 : 0,cc=xy ? 0 : 2;
        require(fields[rr][6]==1 && fields[1][6]==1 && fields[cc][6]==1);
        require(fields[rr][7]==4 && fields[1][7]==0 && fields[cc][7]==0);
        require(fields[rr][11]==0 && fields[1][11]==0 && fields[cc][11]==4);
        require(fields[rr][16]==1 && fields[1][16]==-1 && fields[cc][16]==1);
        for(auto& field:fields)require(field[12]==0);
    }
    GaussianPass pass;pass.axis=0;pass.weights={.25,.5,.25};pass.border=GaussianBorder::reflect;
    require(structure_tensor_into(input.data(),type,pointers.data(),{5,5},{pass},{GaussianBorder::constant,GaussianBorder::constant}));
    require(fields[0][7]==2 && fields[1][6]==.5 && fields[2][11]==2.5);
    auto second=pass;second.axis=1;
    require(structure_tensor_into(input.data(),type,pointers.data(),{5,5},{pass,second},{GaussianBorder::constant,GaussianBorder::constant}));
    require(fields[0][12]==1.25 && fields[1][12]==0 && fields[2][12]==1.25);
}
template<class T> void ramp(std::size_t n,std::size_t side){
    using namespace skimagexx;
    std::vector<std::size_t> shape(n,side);std::size_t count=1,center=0;
    for(std::size_t axis=0;axis<n;++axis){count*=side;center=center*side+side/2;}
    std::vector<T> image(count);
    for(std::size_t p=0;p<count;++p){auto coordinate=p;std::size_t value=0;
        for(std::size_t axis=n;axis;--axis){value+=(coordinate%side)*axis;coordinate/=side;}
        image[p]=T(value);
    }
    std::vector<std::vector<T>> fields(n*(n+1)/2,std::vector<T>(count)),serial;
    std::vector<void*> pointers;for(auto& field:fields)pointers.push_back(field.data());
    const int worker_counts[]={1
#ifdef _OPENMP
        ,4
#endif
    };
    for(auto workers:worker_counts){
        set_num_threads(workers);
        require(structure_tensor_into(image.data(),std::is_same_v<T,float> ? PixelType::float32 : PixelType::float64,pointers.data(),shape,{},std::vector<GaussianBorder>(n,GaussianBorder::reflect)));
        const auto scale=std::size_t(2) << (2*(n-1));std::size_t field=0;
        for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c)require(fields[field++][center]==T(scale*scale*(r+1)*(c+1)));
        if(workers==1)serial=fields;else require(fields==serial);
    }
    set_num_threads(1);
}
}
int main(){
    using namespace skimagexx;
    require(get_num_threads()==1);
    impulse<float>(PixelType::float32,1.f);impulse<double>(PixelType::float64,1.);
    impulse<unsigned char>(PixelType::uint8,255);impulse<unsigned char>(PixelType::boolean,3);
    for(std::size_t n=2;n<=5;++n){ramp<float>(n,3);ramp<double>(n,3);}
    ramp<float>(5,11);ramp<double>(5,11);
    std::vector<double> input(60);
    for(std::size_t r=0;r<3;++r)for(std::size_t c=0;c<4;++c)for(std::size_t z=0;z<5;++z)input[(r*4+c)*5+z]=double(r+2*c+3*z);
    std::array<std::vector<double>,6> fields;std::array<void*,6> pointers;
    for(std::size_t f=0;f<6;++f){fields[f].resize(60);pointers[f]=fields[f].data();}
    require(structure_tensor_into(input.data(),PixelType::float64,pointers.data(),{3,4,5},{},std::vector<GaussianBorder>(3,GaussianBorder::reflect)));
    const double expected[]={1024,2048,3072,4096,6144,9216};
    for(std::size_t f=0;f<6;++f)require(fields[f][(1*4+1)*5+2]==expected[f]);
    input[0]=std::numeric_limits<double>::quiet_NaN();
    require(!structure_tensor_into(input.data(),PixelType::float64,pointers.data(),{3,4,5},{},std::vector<GaussianBorder>(3,GaussianBorder::reflect)));
    input[0]=0.;
    GaussianPass invalid;invalid.axis=0;invalid.weights={.25,.5,.25};invalid.border=GaussianBorder::constant;
    invalid.cval=std::numeric_limits<double>::quiet_NaN();
    require(!structure_tensor_into(input.data(),PixelType::float64,pointers.data(),{3,4,5},{invalid},std::vector<GaussianBorder>(3,GaussianBorder::reflect)));
    set_num_threads(1);
}
