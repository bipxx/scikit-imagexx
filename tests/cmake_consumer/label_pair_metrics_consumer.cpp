#include "../label_pair_metrics_native_controls.hpp"
#include "../fixtures/label_pair_rand_originals_v1.hpp"
#include <iostream>
#include <string>

std::size_t rand_controls() {
    using namespace skimagexx;using namespace skimagexx::detail;
    std::size_t checks=0;
    const auto check=[&](bool value){++checks;if(!value)throw std::runtime_error("installed Rand control "+std::to_string(checks));};
    const auto initial=get_num_threads();check(initial==1);
    for(int workers:{1,4}) {
        set_num_threads(workers);
        for(const auto& row:label_pair_rand_originals::rows) {
            LabelPairRandTerms terms;
            check(label_pair_rand_terms({row.truth.data(),row.test.data(),row.truth.size(),row.ignored.data(),row.ignored.size(),false},terms));
            check(terms.joint==row.terms[0]&&terms.row==row.terms[1]&&terms.column==row.terms[2]);
            LabelPairScalar64 precision,recall,fscore,error;
            check(label_pair_rand_divide(terms.joint,terms.row,precision));
            check(label_pair_rand_divide(terms.joint,terms.column,recall));
            check(label_pair_rand_fscore(terms,row.alpha,fscore));
            check(label_pair_rand_error(fscore.bits,error));
            check(error.bits==row.words[0]&&precision.bits==row.words[1]&&recall.bits==row.words[2]);
            check(get_num_threads()==workers);
        }
    }
    set_num_threads(initial);check(get_num_threads()==1);return checks;
}

int main(int argc,char** argv) {
    const bool rand_only=argc==2&&std::string(argv[1])=="--rand-only";
    if(argc!=1&&!rand_only)return 2;
    if(!rand_only)std::cout << "installed label-pair controls=" << label_pair_controls::run()
              << "; originals=5; workers=1,4; restored=1; available="
              << skimagexx::has_avx2() << '\n';
    std::cout<<"installed Rand controls="<<rand_controls()<<"; originals=5; workers=1,4; restored=1; available="<<skimagexx::has_avx2()<<'\n';
}
