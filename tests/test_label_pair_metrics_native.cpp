#include "label_pair_metrics_native_controls.hpp"
#include <iostream>
int main(){std::cout<<"label-pair native controls="<<label_pair_controls::run()<<"; originals=5; workers=1,4; restored=1; available="<<skimagexx::has_avx2()<<'\n';std::cout<<"density controls="<<label_pair_controls::density()<<'\n';}
