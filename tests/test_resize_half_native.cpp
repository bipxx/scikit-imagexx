#include "../src/resize_half.hpp"
#include "skimagexx/image.hpp"
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
namespace {
void require(bool ok){if(!ok)throw std::runtime_error("resize half check failed");}
std::uint64_t hash(const std::vector<std::uint16_t>& words) {
    std::uint64_t result=14695981039346656037ULL;
    for(auto word:words){result=(result^(word&255))*1099511628211ULL;result=(result^(word>>8))*1099511628211ULL;}
    return result;
}
void check_clip(std::uint16_t lower,std::uint16_t upper,int policy,std::uint64_t expected) {
    std::vector<std::uint16_t> words(65536);for(std::size_t i=0;i<words.size();++i)words[i]=std::uint16_t(i);
    detail::half_clip_into(words.data(),words.size(),lower,upper,policy);
    if(hash(words)!=expected){std::cerr<<"lower="<<lower<<" upper="<<upper<<" policy="<<policy<<"\n";require(false);}
}
void copy_and_range(std::size_t count) {
    std::vector<std::uint16_t> words(count),copy(count),swapped(count);
    for(std::size_t i=0;i<count;++i){words[i]=std::uint16_t(i);swapped[i]=std::uint16_t((words[i]>>8)|(words[i]<<8));}
    detail::half_copy_into(words.data(),copy.data(),count,false);require(words==copy);
    detail::half_copy_into(swapped.data(),copy.data(),count,true);require(words==copy);
    // An unaligned byte source is legal, and must preserve every payload bit.
    std::vector<unsigned char> unaligned(2*count+1);std::memcpy(unaligned.data()+1,swapped.data(),2*count);
    detail::half_copy_into(unaligned.data()+1,copy.data(),count,true);require(words==copy);
    if(count>=65536) {
        std::uint16_t low=0,high=0;const auto info=detail::half_range_into(words.data(),count,&low,&high);
        require(low==0xfc00 && high==0x7c00 && info.has_value && info.has_nan && info.positive_zero && info.negative_zero);
    }
}
void worker_checks() {
    std::vector<std::uint16_t> words(513*1025),one(words.size()),four(words.size());
    for(std::size_t i=0;i<words.size();++i)words[i]=std::uint16_t(i);
    for(int policy:{0,1})for(auto bounds:{std::pair<std::uint16_t,std::uint16_t>{0xbc00,0x3c00},{0x8000,0},{0x7d01,0x3c00}}) {
        one=four=words;set_num_threads(1);detail::half_clip_into(one.data(),one.size(),bounds.first,bounds.second,policy);
        #ifdef _OPENMP
        set_num_threads(4);
        #endif
        detail::half_clip_into(four.data(),four.size(),bounds.first,bounds.second,policy);require(one==four);
    }
    set_num_threads(1);copy_and_range(words.size());
    #ifdef _OPENMP
    set_num_threads(4);
    #endif
    copy_and_range(words.size());set_num_threads(1);
}
template<class E,class F> void rejects(F action){bool caught=false;try{action();}catch(const E&){caught=true;}require(caught);}
}
int main() {
    require(get_num_threads()==1);
    detail::half_copy_into(nullptr,nullptr,0,true);detail::half_clip_into(nullptr,0,0,0,0);
    require(!detail::half_range_into(nullptr,0,nullptr,nullptr).has_value);
    rejects<std::invalid_argument>([]{detail::half_clip_into(nullptr,0,0,0,2);});
    rejects<std::invalid_argument>([]{detail::half_copy_into(nullptr,nullptr,1,true);});
    rejects<std::overflow_error>([]{detail::half_copy_into(nullptr,nullptr,std::numeric_limits<std::size_t>::max(),false);});
    #include "data/resize_half.inc"
    #include "data/resize_half_legacy.inc"
    for(auto count:std::initializer_list<std::size_t>{1,3,7,15,16,17,31,32,33,65536,65537})copy_and_range(count);
    for(auto word:{std::uint16_t{0},std::uint16_t{0x8000},std::uint16_t{0x7e01},std::uint16_t{0xfc00},std::uint16_t{0x7c00}}) {
        std::vector<std::uint16_t> words(1031,word);std::uint16_t lo=0,hi=0;
        const auto info=detail::half_range_into(words.data(),words.size(),&lo,&hi);
        if(detail::half_nan(word))require(info.has_nan && !info.has_value && detail::half_nan(lo) && detail::half_nan(hi));
        else require(info.has_value && !info.has_nan && lo==word && hi==word);
    }
    worker_checks();std::cout<<"30 NumPy all-65536-half clipping fixtures, copy/range/worker checks passed\n";
}
