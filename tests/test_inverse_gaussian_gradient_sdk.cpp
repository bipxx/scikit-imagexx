#include <skimagexx/inverse_gaussian_gradient.hpp>
#include <skimagexx/image.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void check(bool v){if(!v)throw std::runtime_error("IGG complete SDK check failed");}
template<class T>T read(std::ifstream& f){T v;f.read(reinterpret_cast<char*>(&v),sizeof(v));check(bool(f));return v;}
template<class T>void record(std::ifstream& f,std::size_t bytes,std::size_t origin,
 const std::vector<std::ptrdiff_t>& shape,const std::vector<std::ptrdiff_t>& strides,
 const std::vector<std::size_t>& alpha_shape,const std::vector<std::size_t>& expected_shape,
 const std::vector<double>& sigma,int workers){
    std::vector<T> input(bytes/sizeof(T));if(bytes)f.read(reinterpret_cast<char*>(input.data()),bytes);
    std::size_t na=1,n=1;for(auto d:alpha_shape)na*=d;for(auto d:expected_shape)n*=d;
    std::vector<T> alpha(na),expected(n);f.read(reinterpret_cast<char*>(alpha.data()),na*sizeof(T));if(n)f.read(reinterpret_cast<char*>(expected.data()),n*sizeof(T));check(bool(f));
    const auto before=input;
    skimagexx::CropView view{bytes?input.data()+origin/sizeof(T):nullptr,shape,strides,sizeof(T)};
    auto result=skimagexx::inverse_gaussian_gradient<T>(view,sigma,alpha.data(),alpha_shape,workers);
    check(result.shape==expected_shape&&result.values.size()==expected.size());
    check(!bytes||!std::memcmp(input.data(),before.data(),bytes));
    check(!n||!std::memcmp(result.values.data(),expected.data(),n*sizeof(T)));
    check(result.captured_workers==(workers?workers:skimagexx::get_num_threads()));
}
void saved(const char* path,int workers){
    std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);check(std::string(magic,8)=="IGGSDK01");const auto count=read<std::uint64_t>(f);check(count==50);
    for(std::size_t i=0;i<count;++i){const auto length=read<std::uint64_t>(f),item=read<std::uint64_t>(f),rank=read<std::uint64_t>(f),bytes=read<std::uint64_t>(f),origin=read<std::uint64_t>(f),arank=read<std::uint64_t>(f),sigmas=read<std::uint64_t>(f),orank=read<std::uint64_t>(f);check(length<100&&rank<=8&&arank<=8&&orank<=8&&bytes<=8192);
        std::string name(length,' ');f.read(name.data(),length);std::vector<std::ptrdiff_t> shape(rank),strides(rank);std::vector<std::size_t> alpha_shape(arank),output(orank);std::vector<double> sigma(sigmas);
        for(auto& d:shape)d=read<std::int64_t>(f);for(auto& d:strides)d=read<std::int64_t>(f);for(auto& d:alpha_shape)d=read<std::int64_t>(f);for(auto& d:output)d=read<std::int64_t>(f);for(auto& d:sigma)d=read<double>(f);
        try{if(item==4)record<float>(f,bytes,origin,shape,strides,alpha_shape,output,sigma,workers);else{check(item==8);record<double>(f,bytes,origin,shape,strides,alpha_shape,output,sigma,workers);}}
        catch(const std::exception& e){throw std::runtime_error(name+": "+e.what());}
    }
    check(f.peek()==std::char_traits<char>::eof());
}
}
int main(int argc,char** argv){try{check(argc==2&&skimagexx::get_num_threads()==1);for(int workers:{0,1,4})saved(argv[1],workers);check(skimagexx::get_num_threads()==1);std::cout<<"50 saved complete operations x workers0/1/4 PASS\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
