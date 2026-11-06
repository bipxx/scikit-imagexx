#include "../label_pair_metrics_native_controls.hpp"
#include <iostream>

int main() {
    std::cout << "installed label-pair controls=" << label_pair_controls::run()
              << "; originals=5; workers=1,4; restored=1; available="
              << skimagexx::has_avx2() << '\n';
}
