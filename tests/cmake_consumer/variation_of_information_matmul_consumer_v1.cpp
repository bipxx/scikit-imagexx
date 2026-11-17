// Proposal only: root must admit and wire this owning successor before compile.
// Include the complete frozen owning consumer unchanged and execute it once.
#define main vi_frozen_sdk_main_v1
#include "variation_of_information_enabled_consumer.cpp"
#undef main
#include "variation_of_information_matmul_controls_v1.hpp"

int main() {
    if(int status=vi_frozen_sdk_main_v1())return status;
    try{return vi_matmul_sdk_controls::run();}
    catch(const std::exception& error){std::cerr<<"VI matmul SDK primary: "<<error.what()<<'\n';return 1;}
    catch(...){std::cerr<<"VI matmul SDK primary: nonstandard exception\n";return 1;}
}
