// Differential against the immutable shipping C++ leaf, not original skimage.
#include "skimagexx/label_pair_metrics.hpp"
#include "skimagexx/image.hpp"
#include <cfenv>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace skimagexx;using namespace skimagexx::detail;
namespace skimagexx::detail {bool baseline_counts(const LabelPairInput&,LabelPairCsr&,int);bool baseline_weighted(const LabelPairWeightedInput&,LabelPairCsrStorage&,int);bool baseline_rand(const LabelPairInput&,LabelPairRandTerms&,int);}
namespace {
std::size_t checks=0;void check(bool v){++checks;if(!v)throw std::runtime_error("density full-storage differential failed");}
void same(const LabelPairCsr&a,const LabelPairCsr&b){check(a.rows==b.rows&&a.columns==b.columns&&a.total_samples==b.total_samples&&a.kept_samples==b.kept_samples&&a.indptr==b.indptr&&a.indices==b.indices&&a.normalization_fp_status==b.normalization_fp_status);check(a.data.size()==b.data.size()&&!std::memcmp(a.data.data(),b.data.data(),8*a.data.size()));check(a.data.capacity()==b.data.capacity()&&a.indices.capacity()==b.indices.capacity());}
void same(const LabelPairCsrStorage&a,const LabelPairCsrStorage&b){check(a.rows==b.rows&&a.columns==b.columns&&a.indptr==b.indptr&&a.indices==b.indices);check(a.data.size()==b.data.size()&&!std::memcmp(a.data.data(),b.data.data(),8*a.data.size()));check(a.data.capacity()==b.data.capacity()&&a.indices.capacity()==b.indices.capacity());}
void one(std::size_t n,std::size_t labels,int weight_kind){
 struct Restore{std::fenv_t e;Restore(){std::fegetenv(&e);}~Restore(){std::fesetenv(&e);}}restore;
 std::vector<std::int64_t>a(n),b(n),ignored;std::vector<double>w(n);std::size_t kept=0;
 for(std::size_t i=0;i<n;++i){a[i]=i%labels;b[i]=(i/3+i%7)%labels;kept+=a[i]!=0;}
 if(weight_kind==2)for(std::size_t i=0;i<labels;++i)ignored.push_back(i);else ignored={0,0,-1};
 const double common=kept?1./double(kept):1.;const std::uint64_t nan=UINT64_C(0xfff8123456789abc);double quiet;std::memcpy(&quiet,&nan,8);
 for(std::size_t i=0;i<n;++i)w[i]=weight_kind==2?quiet:weight_kind==1?(i%3?common:0.):a[i]?common:0.;
 const auto before_a=a,before_b=b;const auto before_w=w;
 for(int budget:{1,4,0}){set_num_threads(budget==0?4:1);
  std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO|FE_INEXACT);const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();
  for(bool norm:{false,true}){LabelPairInput in{a.data(),b.data(),n,ignored.data(),ignored.size(),norm};LabelPairCsr old,current;check(baseline_counts(in,old,budget));check(label_pair_counts(in,current,budget));same(old,current);check(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==rounding);
   if(!norm){LabelPairRandTerms old_terms,new_terms;check(baseline_rand(in,old_terms,budget));check(label_pair_rand_terms(in,new_terms,budget));check(!std::memcmp(&old_terms,&new_terms,sizeof(old_terms)));}}
  LabelPairWeightedInput in{a.data(),b.data(),w.data(),n};LabelPairCsrStorage old,current;check(baseline_weighted(in,old,budget));check(label_pair_counts_preweighted(in,current,budget));same(old,current);check(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==rounding);
  check(a==before_a&&b==before_b&&!std::memcmp(w.data(),before_w.data(),n*8));check(get_num_threads()==(budget==0?4:1));
 }
}
}
int main(){set_num_threads(1);for(auto n:{std::size_t{3},std::size_t{4}})one(n,1,0);one(63,4,0);one(64,4,0);one(4095,32,0);one(4096,32,0);one(262144,256,0);one(262144,257,0);one(262144,64,0);one(4096,8,1);one(1024,8,2);one(1048576,1,0);set_num_threads(1);check(get_num_threads()==1);std::cout<<"PASS density_differential_controls="<<checks<<" full_backing_words=exact budgets=1,4,default restored=1\n";}
