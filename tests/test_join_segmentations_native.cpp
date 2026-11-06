#include "cmake_consumer/join_segmentations_consumer.hpp"
#include <iostream>
int main(){
    if(skimagexx::get_num_threads()!=1)throw std::runtime_error("join SDK default worker budget must be1");
    const auto one=skimagexx_test::join_segmentations_consumer(1);
    const auto four=skimagexx_test::join_segmentations_consumer(4);
    std::cout<<"join native checks "<<one<<" "<<four<<"\n";
}
