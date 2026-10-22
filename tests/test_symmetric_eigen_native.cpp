#include "skimagexx/symmetric_eigen.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool good){if(!good)throw std::runtime_error("native symmetric eigenvalue check failed");}
template<class T> void check(std::size_t n,const std::vector<T>& upper,const std::vector<double>& expected,bool pair){
    constexpr std::size_t pixels=1031;
    std::vector<std::vector<T>> fields;
    for(auto value:upper)fields.emplace_back(pixels,value);
    std::vector<const void*> pointers;for(auto& field:fields)pointers.push_back(field.data());
    std::vector<T> output(n*pixels),serial;
    for(bool interleaved:{false,true})for(int workers:{1,4}){
        skimagexx::set_num_threads(workers);
        require(skimagexx::symmetric_eigenvalues_into(pointers.data(),output.data(),pixels,n,sizeof(T)==4,pair,skimagexx::PairPrecision::native,interleaved));
        for(std::size_t p=0;p<pixels;++p)for(std::size_t d=0;d<n;++d){
            const auto value=output[interleaved ? p*n+n-1-d : d*pixels+p];
            require(std::abs(double(value)-expected[d])<(sizeof(T)==4 ? 2e-6 : 2e-13));
        }
        if(workers==1)serial=output;else require(output==serial);
    }
    fields[0][17]=std::numeric_limits<T>::quiet_NaN();
    require(!skimagexx::symmetric_eigenvalues_into(pointers.data(),output.data(),pixels,n,sizeof(T)==4,pair));
}
template<class T> void check_inactive_lane(){
    constexpr std::size_t count=17;
    for(std::size_t diagonal:{0,3,5}){
        std::vector<std::vector<T>> fields(6,std::vector<T>(count));
        fields[diagonal][0]=-T(0);
        for(std::size_t f=0;f<6;++f)for(std::size_t p=1;p<count;++p)fields[f][p]=T((f+1)*p*.01);
        std::vector<const void*> pointers;for(auto& field:fields)pointers.push_back(field.data());
        std::vector<T> output(3*count);
        require(skimagexx::symmetric_eigenvalues_into(pointers.data(),output.data(),count,3,sizeof(T)==4,false));
        const auto negative=diagonal==0 ? 2 : diagonal==3 ? 1 : 0;
        for(std::size_t d=0;d<3;++d){require(output[d*count]==0);require(std::signbit(output[d*count])==(d==negative));}
    }
}
template<class T> void check_large_zero_order(){
    constexpr std::size_t count=17;
    std::vector<T> zeros(count,T(0)),negative(count,-T(0));
    for(std::size_t n:{22,64}){
        std::vector<const void*> pointers(n*(n+1)/2,zeros.data());pointers[0]=negative.data();
        std::vector<T> output(n*count);
        require(skimagexx::symmetric_eigenvalues_into(pointers.data(),output.data(),count,n,sizeof(T)==4,false));
        const auto negative_axis=n==22 ? 0 : 31;
        for(std::size_t d=0;d<n;++d)for(std::size_t p=0;p<count;++p){
            require(output[d*count+p]==0);require(std::signbit(output[d*count+p])==(d==negative_axis));
        }
        pointers[0]=zeros.data();pointers.back()=negative.data();
        require(skimagexx::symmetric_eigenvalues_into(pointers.data(),output.data(),count,n,sizeof(T)==4,false));
        for(std::size_t d=0;d<n;++d)for(std::size_t p=0;p<count;++p){
            require(output[d*count+p]==0);require(std::signbit(output[d*count+p])==(n==22 && d==21));
        }
    }
}
}
int main(){
    require(skimagexx::get_num_threads()==1);
    check<float>(2,{2,1,2},{3,1},true);
    check<double>(2,{2,1,2},{3,1},true);
    check<float>(3,{2,1,0,2,0,5},{5,3,1},false);
    check<double>(3,{2,1,0,2,0,5},{5,3,1},false);
    std::vector<double> eigenvalues;for(int k=1;k<=4;++k)eigenvalues.push_back(2+2*std::cos(k*std::acos(-1.)/5));
    check<double>(4,{2,1,0,0,2,1,0,2,1,2},eigenvalues,false);
    check_inactive_lane<float>();check_inactive_lane<double>();
    check_large_zero_order<float>();check_large_zero_order<double>();
    require(skimagexx::symmetric_eigenvalues_into(nullptr,nullptr,0,3,false,false));
    bool rejected=false;try{skimagexx::symmetric_eigenvalues_into(nullptr,nullptr,1,0,false);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected);skimagexx::set_num_threads(1);
}
