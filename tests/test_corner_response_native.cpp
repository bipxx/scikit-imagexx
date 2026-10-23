#include "skimagexx/corner_response.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
void require(bool condition){if(!condition)throw std::runtime_error("native corner response check failed");}
template<class T> void check(){
    using namespace skimagexx;constexpr std::size_t count=32939;
    std::vector<T> a(count,T(3)),b(count,T(1)),c(count,T(2));
    for(std::size_t i=0;i<count;i+=3)a[i]=b[i]=c[i]=0;
    const void* fields[]={a.data(),b.data(),c.data()};
    std::vector<T> first(count),second(count),serial_first,serial_second;
    const int worker_counts[]={1
#ifdef _OPENMP
        ,4
#endif
    };
    for(auto method:{CornerResponse::harris,CornerResponse::noble,CornerResponse::shi_tomasi,CornerResponse::foerstner}){
        for(auto workers:worker_counts){
            set_num_threads(workers);
            const auto parameter=method==CornerResponse::harris ? .05 : 1.;
            require(corner_response_into(fields,first.data(),method==CornerResponse::foerstner ? second.data() : nullptr,count,std::is_same_v<T,float>,method,parameter));
            const double expected=method==CornerResponse::harris ? 3.75 : method==CornerResponse::noble ? 10./6 : method==CornerResponse::shi_tomasi ? (5-std::sqrt(5.))/2 : 1.;
            for(std::size_t i=0;i<count;++i){
                require(std::abs(double(first[i])-(i%3 ? expected : 0.))<(sizeof(T)==4 ? 2e-6 : 2e-13));
                if(method==CornerResponse::foerstner)require(std::abs(double(second[i])-(i%3 ? .8 : 0.))<(sizeof(T)==4 ? 2e-6 : 2e-13));
            }
            if(workers==1){serial_first=first;serial_second=second;}
            else{require(first==serial_first);if(method==CornerResponse::foerstner)require(second==serial_second);}
        }
    }
    require(!corner_response_into(fields,first.data(),nullptr,count,std::is_same_v<T,float>,CornerResponse::noble,0));
    a[17]=std::numeric_limits<T>::quiet_NaN();
    require(!corner_response_into(fields,first.data(),nullptr,count,std::is_same_v<T,float>,CornerResponse::harris,.05));
    set_num_threads(1);
}
}
int main(){
    using namespace skimagexx;require(get_num_threads()==1);check<float>();check<double>();
    std::array<float,17> a,b,c; a.fill(3);b.fill(1);c.fill(2);const void* fields[]={a.data(),b.data(),c.data()};
    std::array<double,17> output;
    require(corner_response_into(fields,output.data(),nullptr,17,true,CornerResponse::harris,.123456789012345,true));
    for(auto value:output)require(value==5.-.123456789012345*25.);
    require(corner_response_into(nullptr,nullptr,nullptr,0,false,CornerResponse::harris,.05));
}
