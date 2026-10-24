#include "skimagexx/max_tree.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

using namespace skimagexx;
namespace {
std::size_t checks=0,source_rows=0;
std::string context;
void require(bool condition,const char* message) {
    ++checks;if(!condition)throw std::runtime_error(context+": "+message);
}
template<class T> bool equal(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && (!a.size() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
template<class T> T read(std::ifstream& file) {
    T value{};file.read(reinterpret_cast<char*>(&value),sizeof(value));
    require(bool(file),"truncated oracle");return value;
}
template<class T> std::vector<T> words(std::ifstream& file,std::size_t count) {
    std::vector<T> result(count);file.read(reinterpret_cast<char*>(result.data()),std::streamsize(count*sizeof(T)));
    require(bool(file),"truncated oracle words");return result;
}
template<class T> struct Arena {
    std::vector<T> image,out,inverted;
    std::vector<std::int64_t> parent,traverser,roots,scratch,minimum,maximum,source_parent,source_traverser;
    std::vector<double> attribute;
    std::vector<std::uint64_t> labels;
    MaxTreeImage input;
    MaxTreeParameters params;
    explicit Arena(std::vector<T> values,MaxTreeType type,MaxTreeOperation operation,
                   std::size_t rank,std::array<std::size_t,3> shape,int connectivity=1,double threshold=5):
        image(std::move(values)),out(image.size(),T(7)),inverted(image.size(),T(9)),
        parent(image.size(),-7),traverser(image.size(),-8),roots(image.size(),-9),scratch(image.size(),-10),
        minimum(image.size()*rank,-11),maximum(image.size()*rank,-12),attribute(image.size(),-13),labels(image.size(),14),
        input{image.data(),type,image.size(),rank,shape},params{operation,connectivity,threshold} {}
    MaxTreeWorkspace workspace() {
        return {{parent.data(),parent.size()},{traverser.data(),traverser.size()},{roots.data(),roots.size()},{scratch.data(),scratch.size()},
            {inverted.data(),inverted.size()},{attribute.data(),attribute.size()},{minimum.data(),minimum.size()},{maximum.data(),maximum.size()}};
    }
    MaxTreeBuffer output() {
        if(params.operation==MaxTreeOperation::tree)return {nullptr,0};
        if(params.operation==MaxTreeOperation::local_maxima)return {labels.data(),labels.size()};
        return {out.data(),out.size()};
    }
    MaxTreeProvided supplied(bool reuse) {
        return reuse?MaxTreeProvided{{source_parent.data(),source_parent.size()},{source_traverser.data(),source_traverser.size()}}:MaxTreeProvided{{nullptr,0},{nullptr,0}};
    }
    auto snapshot() const {
        std::vector<unsigned char> data;
        auto append=[&](const auto& values) {
            if(values.empty())return;
            const auto* bytes=reinterpret_cast<const unsigned char*>(values.data());
            data.insert(data.end(),bytes,bytes+values.size()*sizeof(typename std::decay_t<decltype(values)>::value_type));
        };
        append(image);append(out);append(inverted);append(parent);append(traverser);append(roots);append(scratch);
        append(minimum);append(maximum);append(attribute);append(labels);append(source_parent);append(source_traverser);return data;
    }
};
template<class T> void source_case(std::ifstream& file,MaxTreeType type,MaxTreeOperation op,
    std::size_t rank,std::array<std::size_t,3> shape,int connectivity,double threshold,bool reuse,std::size_t n,int workers) {
    auto image=words<T>(file,n);auto parent=words<std::int64_t>(file,n),traverser=words<std::int64_t>(file,n);
    std::vector<T> expected;std::vector<std::uint64_t> labels;
    if(op==MaxTreeOperation::local_maxima)labels=words<std::uint64_t>(file,n);
    else if(op!=MaxTreeOperation::tree)expected=words<T>(file,n);
    Arena<T> a(image,type,op,rank,shape,connectivity,threshold);
    if(reuse){a.source_parent=parent;a.source_traverser=traverser;}
    auto old_parent=a.parent,old_order=a.traverser,old_roots=a.roots,old_scratch=a.scratch;
    auto w=a.workspace();auto output=a.output();auto supplied=a.supplied(reuse);
    require(max_tree_ineligibility(a.input,a.params,w,output,supplied)==nullptr,"original safe case rejected");
    auto result=max_tree_into(a.input,a.params,w,output,supplied);
    require(result.tree_reused==reuse,"provided tree status");
    require(result.preparation_workers==1,"small-case preparation team");
    require(equal(a.image,image),"source mutated");
    if(reuse) {
        require(equal(a.source_parent,parent) && equal(a.source_traverser,traverser),"provided tree mutated");
        require(equal(a.parent,old_parent) && equal(a.traverser,old_order) && equal(a.roots,old_roots) && equal(a.scratch,old_scratch),"inactive tree workspace accessed");
    } else require(equal(a.parent,parent) && equal(a.traverser,traverser),"parent/traverser raw-word mismatch");
    if(op==MaxTreeOperation::local_maxima)require(equal(a.labels,labels),"local-max label mismatch");
    else if(op!=MaxTreeOperation::tree)require(equal(a.out,expected),"filtered image raw-word mismatch");
    require(get_num_threads()==workers,"global worker limit changed");++source_rows;
}
#define TYPES(F) \
    F(u8,std::uint8_t) F(u16,std::uint16_t) F(u32,std::uint32_t) F(u64,std::uint64_t) \
    F(i8,std::int8_t) F(i16,std::int16_t) F(i32,std::int32_t) F(i64,std::int64_t) F(f32,float) F(f64,double)
void original_cases(const char* path,int workers) {
    std::ifstream file(path,std::ios::binary);char magic[8]{};file.read(magic,8);
    require(std::memcmp(magic,"MTREEV1\0",8)==0,"oracle magic");
    auto count=read<std::uint32_t>(file);require(count==233,"only233safe source IDs admitted");
    for(std::uint32_t i=0;i<count;++i) {
        auto length=read<std::uint32_t>(file);require(length<512,"oracle ID length");
        context.resize(length);file.read(context.data(),length);
        auto type=MaxTreeType(read<std::uint32_t>(file));auto op=MaxTreeOperation(read<std::uint32_t>(file));
        auto rank=read<std::uint32_t>(file);auto connectivity=read<std::uint32_t>(file);
        std::array<std::size_t,3> shape{};for(auto& dim:shape)dim=std::size_t(read<std::uint64_t>(file));
        auto threshold=read<double>(file);bool reuse=read<std::uint32_t>(file)!=0;auto n=std::size_t(read<std::uint64_t>(file));
        require(n<=512 && (rank==2 || rank==3),"bounded source oracle shape");
        switch(type) {
#define CASE(name,T) case MaxTreeType::name:source_case<T>(file,type,op,rank,shape,int(connectivity),threshold,reuse,n,workers);break;
            TYPES(CASE)
#undef CASE
            default:require(false,"unknown oracle dtype");
        }
    }
    char extra;file.read(&extra,1);require(file.eof(),"oracle trailing data");
}
template<class T> T inverse(T value) {
    if constexpr(std::is_floating_point_v<T>)return T(T(1)-value);
    else if constexpr(std::is_unsigned_v<T>)return T(std::numeric_limits<T>::max()-value);
    else return T(T(-1)-value);
}
template<class T> void independent_ramps(MaxTreeType type,int workers) {
    for(auto shape:{std::array<std::size_t,3>{{3,5,1}},{{4,8,1}},{{3,11,1}},{{5,13,1}},{{3,4,5}}}) {
        const auto rank=shape[2]==1?std::size_t(2):std::size_t(3);const auto n=shape[0]*shape[1]*shape[2];
        for(std::size_t connectivity=1;connectivity<=rank;++connectivity)for(int operation=0;operation<6;++operation) {
            std::vector<T> image(n);for(std::size_t i=0;i<n;++i)image[i]=T(i);
            auto op=MaxTreeOperation(operation);const bool closed=op==MaxTreeOperation::area_closing || op==MaxTreeOperation::diameter_closing;
            Arena<T> a(image,type,op,rank,shape,int(connectivity));auto w=a.workspace();
            auto result=max_tree_into(a.input,a.params,w,a.output());require(result.preparation_workers==1,"tiny independent team");
            for(std::size_t i=0;i<n;++i) {
                require(a.traverser[i]==std::int64_t(closed?n-1-i:i),"stable ramp traversal");
                auto expected=closed?std::min(i+1,n-1):(i?i-1:0);
                require(a.parent[i]==std::int64_t(expected),"ramp parent chain");
                if(op==MaxTreeOperation::local_maxima)require(a.labels[i]==std::uint64_t(i==n-1),"ramp maximum label");
                else if(op!=MaxTreeOperation::tree)require(a.out[i]==T(closed?std::max(i,std::size_t(4)):std::min(i,n-5)),"ramp direct-filter definition");
            }
            if(op!=MaxTreeOperation::tree) {
                a.source_parent=a.parent;a.source_traverser=a.traverser;
                auto expected=a.out;auto expected_labels=a.labels;
                std::fill(a.out.begin(),a.out.end(),T(7));std::fill(a.labels.begin(),a.labels.end(),14);
                auto parameters=a.params;parameters.connectivity=-17; // source ignores connectivity when both tree arrays exist
                auto supplied_workspace=w;
                supplied_workspace.parent=supplied_workspace.traverser=supplied_workspace.roots=supplied_workspace.sort_scratch={nullptr,0};
                auto again=max_tree_into(a.input,parameters,supplied_workspace,a.output(),a.supplied(true));
                require(again.tree_reused,"independent actual tree reuse");
                require(op==MaxTreeOperation::local_maxima?equal(a.labels,expected_labels):equal(a.out,expected),"reused-tree result differs");
            }
            require(get_num_threads()==workers,"independent limit changed");
        }
    }
    context="constant-root";
    std::vector<T> image(33,T(2));
    for(int operation=0;operation<6;++operation) {
        Arena<T> a(image,type,MaxTreeOperation(operation),2,{{3,11,1}},1,34);
        max_tree_into(a.input,a.params,a.workspace(),a.output());
        const bool closed=operation==3 || operation==5;
        for(std::size_t i=0;i<image.size();++i) {
            require(a.parent[i]==0 && a.traverser[i]==std::int64_t(i),"flat stable root identity");
            if(operation==1)require(a.labels[i]==1,"flat maximum labels remain ones");
            else if(operation)require(a.out[i]==(closed?inverse(T(0)):T(0)),"root below threshold literalzero");
        }
    }
}
template<class T> void exact_key_controls(MaxTreeType type,bool high) {
    context="exact-integer-sort-keys";
    std::vector<T> values(20);
    for(std::size_t i=0;i<values.size();++i)
        values[i]=high?T(std::numeric_limits<T>::max()-T(i%4)):T(std::numeric_limits<T>::lowest()+T(i%4));
    Arena<T> a(values,type,MaxTreeOperation::tree,2,{{4,5,1}});
    max_tree_into(a.input,a.params,a.workspace(),a.output());
    std::size_t position=0;
    for(int bucket=0;bucket<4;++bucket)for(std::size_t i=0;i<values.size();++i)
        if(int(i%4)==(high?3-bucket:bucket))require(a.traverser[position++]==std::int64_t(i),"adjacent64-bit limits must not round through float");
}
template<class T> void exact_float_keys(MaxTreeType type) {
    context="exact-float-sort-keys";
    T levels[4]={T(1),T(1),T(1),T(1)};
    for(int i=1;i<4;++i)levels[i]=std::nextafter(levels[i-1],std::numeric_limits<T>::infinity());
    std::vector<T> values(20);for(std::size_t i=0;i<values.size();++i)values[i]=levels[3-i%4];
    Arena<T> a(values,type,MaxTreeOperation::tree,2,{{4,5,1}});
    max_tree_into(a.input,a.params,a.workspace(),a.output());
    std::size_t position=0;
    for(int bucket=0;bucket<4;++bucket)for(std::size_t i=0;i<values.size();++i)
        if(int(i%4)==3-bucket)require(a.traverser[position++]==std::int64_t(i),"adjacentfloat ULP keys must retain native precision");
    context="signed-zero-stable-canonicalization";
    for(std::size_t i=0;i<values.size();++i)values[i]=i%2?T(0):-T(0);
    Arena<T> zeros(values,type,MaxTreeOperation::area_opening,2,{{4,5,1}},1,1);
    max_tree_into(zeros.input,zeros.params,zeros.workspace(),zeros.output());
    const T negative_zero=-T(0);
    for(std::size_t i=0;i<values.size();++i) {
        require(zeros.parent[i]==0 && zeros.traverser[i]==std::int64_t(i),"signedzero keys stable tieorder");
        require(std::memcmp(&zeros.out[i],&negative_zero,sizeof(T))==0,"canonical negativezero propagation rawbits");
    }
}
void guard_controls() {
    context="prewrite-guards";
    for(int fault=0;fault<24;++fault) {
        Arena<float> a(std::vector<float>(20,.25f),MaxTreeType::f32,MaxTreeOperation::area_opening,2,{{4,5,1}});
        auto image=a.input;auto params=a.params;auto w=a.workspace();auto out=a.output();auto provided=a.supplied(false);
        if(fault==0)image.rank=1;
        if(fault==1)image.shape[0]=2;
        if(fault==2)image.shape[0]=std::size_t(std::numeric_limits<std::int32_t>::max())+1;
        if(fault==3)--image.elements;
        if(fault==4)image.type=MaxTreeType(99);
        if(fault==5)params.connectivity=0;
        if(fault==6)params.connectivity=3;
        if(fault==7)params.operation=MaxTreeOperation(99);
        if(fault==8)out.elements=1;
        if(fault==9)out.data=nullptr;
        if(fault==10)w.roots.data=w.parent.data;
        if(fault==11)w.sort_scratch.elements=1;
        if(fault==12)out.data=a.image.data();
        if(fault==13)w.parent.data=reinterpret_cast<std::int64_t*>(reinterpret_cast<unsigned char*>(a.parent.data())+1);
        if(fault==14)image.data=reinterpret_cast<const void*>(std::numeric_limits<std::uintptr_t>::max()-1);
        if(fault==15)a.image[0]=std::numeric_limits<float>::infinity();
        if(fault>=16) {
            a.source_parent.assign(20,0);a.source_traverser.resize(20);for(int i=0;i<20;++i)a.source_traverser[std::size_t(i)]=i;
            provided=a.supplied(true);
            if(fault==16)a.source_parent[1]=99;
            if(fault==17)provided.traverser.elements=19;
            if(fault==18)std::swap(a.source_traverser[0],a.source_traverser[1]);
            if(fault==19)a.source_traverser[1]=0;
            if(fault==20)a.source_parent[0]=1;
            if(fault==21)a.source_parent[2]=2;
            if(fault==22)provided.parent.data=nullptr;
            if(fault==23)params.operation=MaxTreeOperation::tree;
        }
        auto before=a.snapshot();
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        require(max_tree_ineligibility(image,params,w,out,provided)!=nullptr,"invalid metadata accepted");
        require(std::fetestexcept(FE_ALL_EXCEPT)==flags,"guard altered FP flags");
        bool rejected=false;try{max_tree_into(image,params,w,out,provided);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"into did not reject invalid metadata");require(a.snapshot()==before,"rejected call wrote a buffer");
        require(std::fetestexcept(FE_ALL_EXCEPT)==flags,"rejected call altered FP flags");std::feclearexcept(FE_ALL_EXCEPT);
    }
    Arena<float> a(std::vector<float>(20,.25f),MaxTreeType::f32,MaxTreeOperation::area_opening,2,{{4,5,1}});
    auto huge=a.input;huge.shape={{3,std::size_t(std::numeric_limits<std::int32_t>::max()),1}};
    huge.elements=3*huge.shape[1];auto diagonal=a.params;diagonal.connectivity=2;
    require(std::strcmp(max_tree_ineligibility(huge,diagonal,a.workspace(),a.output()),"max-tree raveled neighborhood overflows source int32")==0,"raveled int32 offset overflow must precede memory reads");
    huge.rank=3;huge.shape={{3,50000,50000}};huge.elements=std::size_t(3)*50000*50000;diagonal.connectivity=1;
    require(std::strcmp(max_tree_ineligibility(huge,diagonal,a.workspace(),a.output()),"max-tree raveled neighborhood overflows source int32")==0,"3D stride overflows int32 before memory reads");
    for(int fault=0;fault<9;++fault) {
        auto w=a.workspace();auto params=a.params;auto out=a.output();
        if(fault<=2) {
            params.operation=MaxTreeOperation::area_closing;
            if(fault==0)w.inverted.data=nullptr;
            if(fault==1)--w.inverted.elements;
            if(fault==2)w.inverted.data=a.out.data();
        }else if(fault<=5) {
            params.operation=MaxTreeOperation::diameter_opening;
            if(fault==3)--w.minimum_coordinates.elements;
            if(fault==4)w.maximum_coordinates.data=w.minimum_coordinates.data;
            if(fault==5)w.maximum_coordinates.data=nullptr;
        }else if(fault==6)w.attribute.data=nullptr;
        else if(fault==7)--w.attribute.elements;
        else params.operation=MaxTreeOperation::tree;
        auto before=a.snapshot();
        require(max_tree_ineligibility(a.input,params,w,out)!=nullptr,"missing/aliased active workspace accepted");
        bool rejected=false;try{max_tree_into(a.input,params,w,out);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected && before==a.snapshot(),"active workspace rejection must precede writes");
    }
    auto before=a.snapshot();auto old=std::fegetround();std::fesetround(FE_DOWNWARD);
    require(max_tree_ineligibility(a.input,a.params,a.workspace(),a.output())!=nullptr,"non-nearest accepted");
    require(std::fegetround()==FE_DOWNWARD && a.snapshot()==before,"rounding guard changed state");std::fesetround(old);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    const auto csr=_mm_getcsr();_mm_setcsr(csr|(1u<<15)|(1u<<6));
    require(max_tree_ineligibility(a.input,a.params,a.workspace(),a.output())!=nullptr,"FTZ/DAZ accepted");
    require(_mm_getcsr()==(csr|(1u<<15)|(1u<<6)),"guard altered MXCSR");_mm_setcsr(csr);
#endif
}
void large_team(int workers) {
    context="large-independent-team";
    const std::size_t rows=513,cols=519,n=rows*cols;
    Arena<std::uint8_t> a(std::vector<std::uint8_t>(n,0),MaxTreeType::u8,MaxTreeOperation::area_closing,2,{{rows,cols,1}},1,1);
    auto result=max_tree_into(a.input,a.params,a.workspace(),a.output());
    require(result.preparation_workers==workers,"actual independent team");
    for(std::size_t i=0;i<n;++i)require(a.out[i]==0 && a.parent[i]==0 && a.traverser[i]==std::int64_t(i),"large flat identity/tail");
}
template<class T> struct Canary {
    std::vector<T> values;
    explicit Canary(std::size_t count):values(count+2,T(37)) {}
    T* data(){return values.data()+1;}
    std::size_t size() const{return values.size()-2;}
    MaxTreeSpan<T> span(){return {data(),size()};}
    void check() const{require(values.front()==T(37) && values.back()==T(37),"integer sort/workspace canary overwritten");}
};
template<class T> void integer_sort_controls(MaxTreeType type,int workers) {
    for(auto shape:{std::array<std::size_t,3>{{3,3,1}},{{3,31,1}},{{3,32,1}},{{3,33,1}},
                   {{3,255,1}},{{3,256,1}},{{3,257,1}},{{4,5,7}},{{257,257,1}},{{513,519,1}}}) {
        const auto rank=shape[2]==1?std::size_t(2):std::size_t(3),n=shape[0]*shape[1]*shape[2];
        for(int pattern=0;pattern<3;++pattern) {
            std::vector<T> values(n);
            constexpr std::uint16_t edges[]={0,1,254,255,256,257,511,512,65279,65280,65534,65535};
            for(std::size_t i=0;i<n;++i) {
                if(pattern==0)values[i]=T(std::numeric_limits<T>::max());
                else if(pattern==1)values[i]=T(edges[(i*7+i/13)%12]);
                else values[i]=T((i*40503u+17u)&65535u);
            }
            for(int operation=0;operation<6;++operation) {
                // Large rows exercise every key and ties without multiplying
                // all filter costs; all six routes retain moderate controls.
                if(n>100000 && operation!=0 && operation!=3)continue;
                context="integer-sort-"+std::to_string(sizeof(T))+"/"+std::to_string(n)+"/"+std::to_string(pattern)+"/"+std::to_string(operation);
                const auto op=MaxTreeOperation(operation);const bool closed=operation==3 || operation==5;
                Arena<T> a(values,type,op,rank,shape,1,11);
                std::vector<std::int64_t> expected(n);for(std::size_t i=0;i<n;++i)expected[i]=std::int64_t(i);
                std::stable_sort(expected.begin(),expected.end(),[&](auto left,auto right) {
                    const auto x=closed?inverse(values[std::size_t(left)]):values[std::size_t(left)];
                    const auto y=closed?inverse(values[std::size_t(right)]):values[std::size_t(right)];return x<y;
                });
                Canary<std::int64_t> parent(n),order(n),roots(n),scratch(n),minimum(n*rank),maximum(n*rank);
                Canary<T> inverted(n),output(n);Canary<double> attribute(n);Canary<std::uint64_t> labels(n);
                MaxTreeWorkspace w{parent.span(),order.span(),roots.span(),scratch.span(),{inverted.data(),n},
                                   attribute.span(),minimum.span(),maximum.span()};
                MaxTreeBuffer out=operation==0?MaxTreeBuffer{nullptr,0}:operation==1?MaxTreeBuffer{labels.data(),n}:MaxTreeBuffer{output.data(),n};
                const auto result=max_tree_into(a.input,a.params,w,out);
                require(std::equal(expected.begin(),expected.end(),order.data()),"count/radix permutation differs from independent std::stable_sort");
                require(a.image==values && get_num_threads()==workers,"integer sorting changed source/global worker state");
                require(result.preparation_workers==(n>=std::size_t(1<<18)?workers:1),"integer control preparation worker team");
                parent.check();order.check();roots.check();scratch.check();minimum.check();maximum.check();
                inverted.check();output.check();attribute.check();labels.check();
                // Reusing the actual tree with no sort storage proves supplied
                // parent/traverser remain readonly and bypass both new sorts.
                if(operation!=0) {
                    const auto original_parent=parent.values,original_order=order.values;
                    auto reused=w;reused.parent=reused.traverser=reused.roots=reused.sort_scratch={nullptr,0};
                    const MaxTreeProvided supplied{{parent.data(),n},{order.data(),n}};
                    auto parameters=a.params;parameters.connectivity=-99;
                    std::vector<unsigned char> expected_output(n*(operation==1?sizeof(std::uint64_t):sizeof(T)));
                    std::memcpy(expected_output.data(),out.data,expected_output.size());
                    require(max_tree_into(a.input,parameters,reused,out,supplied).tree_reused,"integer supplied tree did not bypass sorting");
                    require(parent.values==original_parent && order.values==original_order,"integer supplied tree changed");
                    require(std::memcmp(expected_output.data(),out.data,expected_output.size())==0,"integer filtered supplied-tree output changed");
                }
            }
        }
    }
}
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"provide one authenticated source oracle");
        require(std::strcmp(max_tree_interface,"max-tree-prepared-v1")==0 && !max_tree_parallel_traversal,"private interface");
        require(get_num_threads()==1,"default worker limit must be1");
        for(int workers:{1,4}) {
            bool accepted=true;try{set_num_threads(workers);}catch(const std::invalid_argument&){accepted=false;}
            if(!accepted){require(workers==4 && get_num_threads()==1,"portable rejects4 andretains1");continue;}
            original_cases(argv[1],workers);
#define CONTROL(name,T) context="independent-" #name;independent_ramps<T>(MaxTreeType::name,workers);
            TYPES(CONTROL)
#undef CONTROL
            exact_key_controls<std::uint64_t>(MaxTreeType::u64,true);
            exact_key_controls<std::int64_t>(MaxTreeType::i64,false);
            exact_key_controls<std::int64_t>(MaxTreeType::i64,true);
            exact_float_keys<float>(MaxTreeType::f32);exact_float_keys<double>(MaxTreeType::f64);
            guard_controls();large_team(workers);
            integer_sort_controls<std::uint8_t>(MaxTreeType::u8,workers);
            integer_sort_controls<std::uint16_t>(MaxTreeType::u16,workers);
        }
        set_num_threads(1);
        std::cout<<"PASS checks="<<checks<<" source_numeric_projections="<<source_rows<<" unique_source_ids=233 metadata_excluded=118 parallel_traversal=false\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<" checks="<<checks<<" source_rows="<<source_rows<<'\n';return 1;}
}
