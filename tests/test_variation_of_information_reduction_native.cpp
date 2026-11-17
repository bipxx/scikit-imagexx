// Same translation unit exposes only private arithmetic to these inert controls.
#include "../src/variation_of_information.cpp"
#include <iostream>
int main() {
    unsigned checked=0;
    for(std::size_t n:{0,1,2,7,8,9,15,16,17,31,32,33,127,128,129,255,256,257,1023}) {
        // Exactly representable closed-form sums, independent of reduction code.
        const double expected=double(n)*double(n? n-1:0)/2.;
        if(skimagexx::minor_sum(n,[](auto i){return double(i);})!=expected)return 1;
        if(skimagexx::source_dot(n,[](auto i){return double(i);},[](auto){return 0.5;})!=expected/2.)return 2;
        if(skimagexx::source_dot(n,[](auto i){return -double(i);},[](auto){return 0.5;})!=-expected/2.)return 3;
        ++checked;
    }
    // A source-independent fused-product witness: (1+2^-27)*(1-2^-27)
    // rounds to 1 separately, but fused with -1 leaves exactly -2^-54.
    // The same coefficient moves between packet and scalar-tail semantics.
    for(std::size_t n:{15,16,17,31,32,33,47,48,49}) {
        const auto left=[](auto i){return i==0?1.:i==16?1.+0x1p-27:i==32?0x1p-54:0.;};
        const auto right=[](auto i){return i==0?-1.:i==16?1.-0x1p-27:i==32?1.:0.;};
        const double expected=n<=16?-1.:n==32?-0x1p-54:0.;
        if(skimagexx::source_dot(n,left,right)!=expected)return 4;
        ++checked;
    }
    std::cout<<"sum_dot_boundary_controls="<<checked<<" originals=0\n";
}
