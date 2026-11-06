#include "mark_boundaries_consumer.hpp"
#include <skimagexx/image.hpp>
#include <cassert>
#include <iostream>
int main(){using namespace skimagexx;assert(get_num_threads()==1);std::size_t count=0;for(int workers:{1,4}){set_num_threads(workers);count+=skimagexx_test::mark_boundaries_consumer(workers);}set_num_threads(1);assert(count>0);std::cout<<"PASS MARK named complete saved SDK operation "<<count<<"\n";}
