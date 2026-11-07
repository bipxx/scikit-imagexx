#include <skimagexx/rolling_ball.hpp>
#include <skimagexx/image.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void check(bool value){if(!value)throw std::runtime_error("rolling_ball SDK contract failure");}
template<class T>T read(std::ifstream& stream){T v{};stream.read(reinterpret_cast<char*>(&v),sizeof(v));check(bool(stream));return v;}
template<class T>void words(std::ifstream& stream,std::size_t backing,std::size_t origin,
                           std::size_t kernel_n,skimagexx::RollingBallGeometry g,double radius,bool safe) {
    std::vector<T> input(backing),kernel(kernel_n);
    if(backing)stream.read(reinterpret_cast<char*>(input.data()),backing*sizeof(T));
    if(kernel_n)stream.read(reinterpret_cast<char*>(kernel.data()),kernel_n*sizeof(T));
    std::size_t n=1;for(auto v:g.shape)n*=std::size_t(v);
    std::vector<T> expected(n);if(n)stream.read(reinterpret_cast<char*>(expected.data()),n*sizeof(T));check(bool(stream));
    const T* image=backing?input.data()+origin:nullptr;
    const auto actual=skimagexx::rolling_ball(image,g,kernel_n?kernel.data():nullptr,radius,safe);
    check(actual.size()==expected.size());
    check(!n||std::memcmp(actual.data(),expected.data(),n*sizeof(T))==0);
}
void saved(const char* path) {
    std::ifstream stream(path,std::ios::binary);char magic[8];stream.read(magic,8);check(std::string(magic,8)=="ROLLSDK1");
    const auto cases=read<std::uint64_t>(stream);check(cases==52);
    for(std::size_t i=0;i<cases;++i) {
        const auto length=read<std::uint64_t>(stream),itemsize=read<std::uint64_t>(stream),rank=read<std::uint64_t>(stream),
                   backing=read<std::uint64_t>(stream),origin=read<std::uint64_t>(stream),kernel=read<std::uint64_t>(stream);
        const auto radius=read<double>(stream);const bool safe=read<std::uint64_t>(stream)!=0;
        check(length<100&&rank>=1&&rank<=8&&backing<=64&&kernel<=201);
        std::string name(length,' ');stream.read(name.data(),std::streamsize(length));
        skimagexx::RollingBallGeometry g;g.shape.resize(rank);g.strides.resize(rank);g.kernel_shape.resize(rank);
        for(auto& x:g.shape)x=std::ptrdiff_t(read<std::int64_t>(stream));
        for(auto& x:g.strides)x=std::ptrdiff_t(read<std::int64_t>(stream));
        for(auto& x:g.kernel_shape)x=std::ptrdiff_t(read<std::int64_t>(stream));
        try {
            if(itemsize==4)words<float>(stream,backing,origin,kernel,g,radius,safe);
            else{check(itemsize==8);words<double>(stream,backing,origin,kernel,g,radius,safe);}
        }catch(const std::exception& e){throw std::runtime_error(name+": "+e.what());}
    }
    check(stream.peek()==std::char_traits<char>::eof());
}
void independent_controls() {
    // Large radius0 identity crosses the actual parallel threshold. No timing.
    std::vector<float> input(512*512);
    for(std::size_t i=0;i<input.size();++i)input[i]=float(i%17);
    skimagexx::RollingBallGeometry g{{512,512},{512*4,4},{}};
    auto out=skimagexx::rolling_ball(input.data(),g,static_cast<const float*>(nullptr),0.);
    check(out==input&&out.data()!=input.data());
    // NaN-safe source stops only after completing the current kernel row.
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double image[4]={1.,nan,2.,3.},kernel[4]={0.,0.,0.,0.};
    skimagexx::RollingBallGeometry custom{{2,2},{16,8},{2,2}};
    auto safe=skimagexx::rolling_ball(image,custom,kernel,1.,true);
    check(safe.size()==4);
    bool refused=false;try{auto bad=g;bad.kernel_shape={0,3};skimagexx::rolling_ball(input.data(),bad,input.data());}
    catch(const std::invalid_argument&){refused=true;}check(refused);
}
}
int main(int argc,char** argv) {
    try {
        check(argc==2);
        check(skimagexx::get_num_threads()==1);
        saved(argv[1]);independent_controls();
        skimagexx::set_num_threads(4);saved(argv[1]);independent_controls();
        skimagexx::set_num_threads(1);
        std::cout<<"52 saved completeoperations x workers1/4 + large parallel/default/native controls PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
