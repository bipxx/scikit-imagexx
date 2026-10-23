/*
Pinned scikit-image 0.26.0 n-sphere count tables.
Copyright: 2009-2022 the scikit-image team
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#include "skimagexx/footprint_decomposition.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace skimagexx {
namespace {
// 2D uint8 table SHA256: e70b7a367e26c854b2c76b88b7b74bd34c6c9d5534de7f3aa1f23421414c2f87
constexpr std::uint8_t sphere2[251][3]={
    {0,0,0},
    {0,0,1},
    {0,1,1},
    {0,2,1},
    {0,2,2},
    {0,3,2},
    {1,1,1},
    {1,2,1},
    {2,0,0},
    {1,3,2},
    {1,4,2},
    {2,2,1},
    {2,3,1},
    {3,1,0},
    {2,4,2},
    {2,5,2},
    {3,3,1},
    {2,6,3},
    {4,2,0},
    {3,5,2},
    {3,6,2},
    {4,4,1},
    {3,7,3},
    {4,5,2},
    {4,6,2},
    {5,4,1},
    {5,5,1},
    {4,8,3},
    {5,6,2},
    {5,7,2},
    {6,5,1},
    {6,6,1},
    {5,9,3},
    {6,7,2},
    {6,8,2},
    {7,6,1},
    {6,9,3},
    {6,10,3},
    {7,8,2},
    {7,9,2},
    {7,9,3},
    {7,10,3},
    {7,11,3},
    {8,9,2},
    {8,10,2},
    {8,10,3},
    {8,11,3},
    {8,12,3},
    {9,10,2},
    {8,13,4},
    {9,11,3},
    {9,12,3},
    {10,10,2},
    {10,11,2},
    {9,14,4},
    {10,12,3},
    {10,13,3},
    {11,11,2},
    {11,12,2},
    {10,15,4},
    {11,13,3},
    {11,14,3},
    {12,12,2},
    {12,13,2},
    {11,16,4},
    {12,14,3},
    {12,15,3},
    {12,15,4},
    {12,16,4},
    {12,17,4},
    {13,15,3},
    {13,16,3},
    {13,16,4},
    {13,17,4},
    {14,15,3},
    {14,16,3},
    {14,17,3},
    {14,17,4},
    {14,18,4},
    {15,16,3},
    {14,19,5},
    {14,20,5},
    {15,18,4},
    {15,19,4},
    {16,17,3},
    {15,20,5},
    {15,21,5},
    {16,19,4},
    {16,20,4},
    {17,18,3},
    {16,21,5},
    {17,19,4},
    {17,20,4},
    {16,23,6},
    {18,19,3},
    {17,22,5},
    {18,20,4},
    {18,21,4},
    {17,24,6},
    {18,22,5},
    {18,23,5},
    {19,21,4},
    {19,22,4},
    {18,25,6},
    {19,23,5},
    {19,24,5},
    {20,22,4},
    {20,23,4},
    {20,23,5},
    {20,24,5},
    {20,25,5},
    {21,23,4},
    {20,26,6},
    {21,24,5},
    {21,25,5},
    {21,26,5},
    {22,24,4},
    {21,27,6},
    {22,25,5},
    {22,26,5},
    {22,27,5},
    {22,27,6},
    {22,28,6},
    {23,26,5},
    {23,27,5},
    {24,25,4},
    {23,28,6},
    {23,29,6},
    {24,27,5},
    {24,28,5},
    {24,28,6},
    {24,29,6},
    {24,30,6},
    {25,28,5},
    {24,31,7},
    {25,29,6},
    {25,30,6},
    {25,31,6},
    {26,29,5},
    {26,30,5},
    {26,30,6},
    {26,31,6},
    {26,32,6},
    {26,32,7},
    {26,33,7},
    {27,31,6},
    {27,32,6},
    {26,35,8},
    {27,33,7},
    {27,34,7},
    {28,32,6},
    {28,33,6},
    {29,31,5},
    {28,34,7},
    {28,35,7},
    {29,33,6},
    {28,36,8},
    {30,32,5},
    {29,35,7},
    {29,36,7},
    {30,34,6},
    {29,37,8},
    {30,35,7},
    {30,36,7},
    {30,37,7},
    {30,37,8},
    {30,38,8},
    {31,36,7},
    {31,37,7},
    {30,40,9},
    {32,36,6},
    {31,39,8},
    {32,37,7},
    {32,38,7},
    {31,41,9},
    {32,39,8},
    {32,40,8},
    {33,38,7},
    {32,41,9},
    {33,39,8},
    {33,40,8},
    {33,41,8},
    {34,39,7},
    {34,40,7},
    {34,40,8},
    {34,41,8},
    {35,39,7},
    {35,40,7},
    {34,43,9},
    {35,41,8},
    {35,42,8},
    {34,45,10},
    {36,41,7},
    {35,44,9},
    {36,42,8},
    {36,43,8},
    {37,41,7},
    {36,44,9},
    {36,45,9},
    {37,43,8},
    {37,44,8},
    {38,42,7},
    {37,45,9},
    {38,43,8},
    {38,44,8},
    {38,45,8},
    {38,45,9},
    {38,46,9},
    {39,44,8},
    {39,45,8},
    {38,48,10},
    {39,46,9},
    {39,47,9},
    {40,45,8},
    {40,46,8},
    {39,49,10},
    {40,47,9},
    {40,48,9},
    {41,46,8},
    {40,49,10},
    {40,50,10},
    {41,48,9},
    {40,51,11},
    {42,47,8},
    {41,50,10},
    {42,48,9},
    {42,49,9},
    {42,50,9},
    {42,50,10},
    {42,51,10},
    {43,49,9},
    {43,50,9},
    {42,53,11},
    {43,51,10},
    {43,52,10},
    {44,50,9},
    {44,51,9},
    {43,54,11},
    {44,52,10},
    {44,53,10},
    {45,51,9},
    {44,54,11},
    {44,55,11},
    {45,53,10},
    {45,54,10},
    {46,52,9},
    {45,55,11},
    {45,56,11},
    {46,54,10},
    {46,55,10},
    {46,55,11},
};
// 3D uint8 table SHA256: 0bde98c3cda6da86fa7267b285f2b370d5495d9332348593f4c963123796ccb5
constexpr std::uint8_t sphere3[101][3]={
    {0,0,0},
    {0,0,1},
    {0,1,1},
    {0,2,1},
    {0,3,1},
    {0,3,2},
    {0,4,2},
    {0,4,3},
    {0,5,3},
    {1,3,0},
    {1,4,0},
    {0,7,4},
    {1,5,1},
    {1,6,1},
    {0,9,5},
    {1,7,2},
    {1,8,2},
    {0,11,6},
    {1,9,3},
    {1,10,3},
    {0,13,7},
    {1,11,4},
    {2,9,1},
    {2,10,1},
    {1,13,5},
    {2,11,2},
    {2,12,2},
    {3,9,0},
    {3,10,0},
    {3,11,0},
    {3,12,0},
    {3,13,0},
    {3,14,0},
    {3,15,0},
    {3,16,0},
    {2,18,5},
    {4,12,0},
    {4,13,0},
    {4,14,0},
    {4,15,0},
    {4,16,0},
    {4,17,0},
    {4,18,0},
    {4,19,0},
    {4,20,0},
    {5,15,0},
    {5,16,0},
    {5,17,0},
    {5,18,0},
    {5,19,0},
    {5,20,0},
    {5,21,0},
    {5,22,0},
    {5,23,0},
    {5,24,0},
    {6,19,0},
    {6,20,0},
    {6,21,0},
    {6,22,0},
    {6,23,0},
    {6,24,0},
    {6,25,0},
    {6,26,0},
    {6,27,0},
    {6,28,0},
    {7,23,0},
    {7,24,0},
    {7,25,0},
    {7,26,0},
    {7,27,0},
    {7,28,0},
    {7,29,0},
    {7,30,0},
    {7,31,0},
    {7,32,0},
    {8,27,0},
    {8,28,0},
    {8,29,0},
    {8,30,0},
    {8,31,0},
    {8,32,0},
    {8,33,0},
    {8,34,0},
    {8,35,0},
    {8,36,0},
    {9,31,0},
    {9,32,0},
    {9,33,0},
    {9,34,0},
    {9,35,0},
    {9,36,0},
    {9,37,0},
    {9,38,0},
    {9,39,0},
    {9,40,0},
    {9,41,0},
    {10,36,0},
    {10,37,0},
    {10,38,0},
    {10,39,0},
    {10,40,0},
};
}
std::array<std::uint8_t,3> footprint_sphere_counts(int dimensions,std::ptrdiff_t radius) {
    const auto size=dimensions==2 ? 251:dimensions==3 ? 101:0;
    if(!size)throw std::invalid_argument("sphere dimensions must be 2 or 3");
    if(radius<0)radius+=size;
    if(radius<0 || radius>=size)throw std::out_of_range("sphere table radius out of bounds");
    const auto* row=dimensions==2 ? sphere2[radius]:sphere3[radius];
    return {row[0],row[1],row[2]};
}
std::vector<FootprintCross> footprint_cross_sums(const std::ptrdiff_t* sums,std::size_t count,std::size_t height) {
    if(count && !sums)throw std::invalid_argument("nonnull cross sums required");
    const auto max=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(count>max || height>max)throw std::overflow_error("cross plan shape overflow");
    for(std::size_t i=0;i<count;++i)if(sums[i]<0 || std::size_t(sums[i])>height || (i && sums[i]>sums[i-1]))throw std::invalid_argument("cross sums must be nonincreasing bounded counts");
    std::vector<FootprintCross> result;std::size_t previous=0;std::ptrdiff_t total=0;
    auto add=[&](std::ptrdiff_t v,std::ptrdiff_t h){
        const auto found=std::find_if(result.begin(),result.end(),[&](const auto& x){return x.vertical==v && x.horizontal==h;});
        if(found==result.end())result.push_back({v,h,1});else ++found->repetitions;
    };
    for(std::size_t i=1;i<count;++i) {
        const auto next=i+1<count ? sums[i+1]:0;
        if(sums[i]>next) {
            const auto vertical=sums[previous]-sums[i];
            add(vertical,std::ptrdiff_t(i-previous));total+=vertical;previous=i;
        }
    }
    const auto remaining=std::ptrdiff_t(height)-1-total;
    if(remaining>0)add(remaining,0);
    return result;
}
std::vector<FootprintCross> footprint_cross_costs(const double* costs,std::size_t count,double limit) {
    if(count && !costs)throw std::invalid_argument("nonnull cross costs required");
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))throw std::overflow_error("cross cost size overflow");
    const auto first=count/2,q=count-first;
    for(std::size_t i=first;i<count;++i)if(!(costs[i]>=0.) || (i>first && costs[i]<costs[i-1]))throw std::invalid_argument("quadrant costs must be nonnegative and nondecreasing");
    std::vector<std::ptrdiff_t> sums(q);
    detail::partition(q,detail::active_workers(count),[&](auto begin,auto end,auto) {
        if(begin==end)return;
        std::size_t low=0,high=q;
        while(low<high){const auto middle=low+(high-low)/2;if(costs[first+middle]+costs[first+begin]<=limit)low=middle+1;else high=middle;}
        auto rows=low;
        for(auto column=begin;column<end;++column) {
            while(rows && !(costs[first+rows-1]+costs[first+column]<=limit))--rows;
            sums[column]=std::ptrdiff_t(rows);
        }
    });
    return footprint_cross_sums(sums.data(),q,q);
}
void footprint_element_into(void* output,std::size_t itemsize,const void* zero,const void* one,int kind,int dimensions,int axis,int direction,std::size_t vertical,std::size_t horizontal) {
    if(kind<0 || kind>2 || (dimensions!=2 && dimensions!=3) || axis<0 || axis>=dimensions || (direction!=0 && direction!=2))throw std::invalid_argument("invalid footprint element parameters");
    const auto max=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(vertical>max/2 || horizontal>max/2)throw std::overflow_error("cross element shape overflow");
    const auto rows=kind==2 ? 2*vertical+1:std::size_t{3};
    const auto columns=kind==2 ? 2*horizontal+1:std::size_t{3};
    if(rows>max/columns)throw std::overflow_error("cross element size overflow");
    const auto count=kind==2 ? rows*columns:dimensions==2 ? std::size_t{9}:std::size_t{27};
    if(itemsize && count>max/itemsize)throw std::overflow_error("cross element byte size overflow");
    if(!itemsize)return;
    if(!output || !zero || !one || output==zero || output==one)throw std::invalid_argument("nonnull disjoint element storage required");
    auto* bytes=static_cast<unsigned char*>(output);
    for(std::size_t i=0;i<count;++i) {
        bool value=false;
        if(kind==2)value=(horizontal && i/columns==vertical) || (vertical && i%columns==horizontal);
        else {
            const int x=int(i%3),y=int((i/3)%3),z=dimensions==3 ? int(i/9):1;
            if(kind==1)value=(x!=1)+(y!=1)+(z!=1)<=1;
            else {
                const int coordinate[3]={dimensions==3 ? z:y,dimensions==3 ? y:x,x};
                bool stem=true;for(int d=0;d<dimensions;++d)if(d!=axis && coordinate[d]!=1)stem=false;
                value=stem || coordinate[axis]==direction;
            }
        }
        std::memcpy(bytes+i*itemsize,value ? one:zero,itemsize);
    }
}
}
