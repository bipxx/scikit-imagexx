#include "cmake_consumer/mark_boundaries_consumer.hpp"
#include <skimagexx/image.hpp>
#include <iostream>
#include <cfenv>
#include <stdexcept>
#ifdef MARK_WRAP_GETTER
std::size_t snapshots=0;
extern "C" int __real__ZN9skimagexx15get_num_threadsEv();
extern "C" int __wrap__ZN9skimagexx15get_num_threadsEv(){++snapshots;return __real__ZN9skimagexx15get_num_threadsEv();}
#endif
int main(){try{
    skimagexx::set_num_threads(1);if(skimagexx::get_num_threads()!=1)throw std::runtime_error("mark defaultglobal1");
    std::size_t checked=0;for(int workers:{1,4}){skimagexx::set_num_threads(workers);checked+=skimagexx_test::mark_boundaries_consumer(workers);if(skimagexx::get_num_threads()!=workers)throw std::runtime_error("mark global state changed");}
    auto c=skimagexx_test::mark_saved_cases().front();skimagexx_test::MarkStorage image(c.image,c.image_bytes),labels(c.labels,c.label_bytes);skimagexx::set_num_threads(1);
#ifdef MARK_WRAP_GETTER
    snapshots=0;
#endif
    auto out=skimagexx::mark_boundaries(image.view,labels.view,c.parameters);skimagexx_test::mark_compare(out,c);
#ifdef MARK_WRAP_GETTER
    if(snapshots!=1)throw std::runtime_error("mark composition must snapshot exactlyonce");
    snapshots=0;out=skimagexx::mark_boundaries(image.view,labels.view,c.parameters,4);if(snapshots)throw std::runtime_error("mark positive worker budget resnapshotted");
#endif
    if(skimagexx::get_num_threads()!=1)throw std::runtime_error("mark restoredefault1");
    std::cout<<"PASS complete mark operation saved="<<checked<<" profiles=2 workers=1,4 default1 owning_lifetime=true\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;} }
