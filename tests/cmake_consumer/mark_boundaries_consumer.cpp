#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__)
#define main mark_wide_saved_main
#include "mark_wide_outer_consumer.cpp"
#undef main
#endif
#include "mark_boundaries_consumer.hpp"
#include <skimagexx/image.hpp>
#include <cassert>
#include <iostream>
namespace {
#ifdef _OPENMP
constexpr int worker_counts[]={1,4};
#else
constexpr int worker_counts[]={1};
#endif
}
int main(){using namespace skimagexx;assert(get_num_threads()==1);std::size_t count=0;for(int workers:worker_counts){set_num_threads(workers);count+=skimagexx_test::mark_boundaries_consumer(workers);}set_num_threads(1);assert(count>0);std::cout<<"PASS MARK named complete saved SDK operation "<<count<<"\n";
#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__)
return mark_wide_saved_main();
#endif
}
