#include "cmake_consumer/regular_grid_consumer.hpp"
#include <iostream>
int main(){
    if(skimagexx::get_num_threads()!=1)throw std::runtime_error("regular_grid SDK default1 required");
    const auto one=skimagexx_test::regular_grid_consumer(1),four=skimagexx_test::regular_grid_consumer(4);
    std::cout<<"regular_grid serial metadata controls "<<one<<" "<<four<<"\n";
}
