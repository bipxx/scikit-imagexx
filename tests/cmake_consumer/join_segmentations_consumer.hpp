#pragma once
#include <skimagexx/join_segmentations.hpp>
#include <skimagexx/component_consumers.hpp>
#include <skimagexx/image.hpp>
#include <cfenv>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "../fixtures/join_segmentations_originals_v1.hpp"

namespace skimagexx_test {
// Append to the existing segmentation installed-SDK family after integration.
// Promotion is explicit SDK metadata; no NumPy or Python provider is required.
inline std::size_t join_segmentations_consumer(int workers) {
    using namespace skimagexx;
    std::size_t checks=0;auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("join SDK control failed");};
    const auto old=get_num_threads();struct Restore{int n;~Restore(){set_num_threads(n);}}restore{old};
    set_num_threads(workers);check(get_num_threads()==workers);
    auto pixel=[](JoinType t){
        if(t.is_signed){switch(t.bytes){case 1:return PixelType::int8;case 2:return PixelType::int16;case 4:return PixelType::int32;default:return PixelType::int64;}}
        switch(t.bytes){case 1:return PixelType::uint8;case 2:return PixelType::uint16;case 4:return PixelType::uint32;default:return PixelType::uint64;}
    };
    auto source=[&](const std::vector<std::uint8_t>& logical,JoinType type,const std::vector<std::size_t>& shape,const std::vector<std::ptrdiff_t>& strides,std::vector<std::uint8_t>& storage){
        std::ptrdiff_t low=0,high=0;for(std::size_t axis=0;axis<shape.size();++axis){auto span=std::ptrdiff_t(shape[axis]-1)*strides[axis];low+=std::min<std::ptrdiff_t>(0,span);high+=std::max<std::ptrdiff_t>(0,span);}
        storage.assign(std::size_t(high-low)+type.bytes+16,0xad);auto* base=storage.data()+8-low;
        for(std::size_t i=0;i<logical.size()/type.bytes;++i){auto index=i;std::ptrdiff_t offset=0;for(auto axis=shape.size();axis-->0;){offset+=std::ptrdiff_t(index%shape[axis])*strides[axis];index/=shape[axis];}std::memcpy(base+offset,logical.data()+i*type.bytes,type.bytes);}
        return ComponentsView{base,pixel(type),false,false,shape,strides};
    };
    for(const auto& row:join_segmentations_originals::rows()) {
        std::vector<std::uint8_t> actual(row.encoded.size());
        join_encode(row.left.data(),row.type,row.right.data(),row.type,row.count,row.factor,row.type,row.type,actual.data());
        check(actual==row.encoded);
        if(!row.full)continue;
        std::vector<std::uint8_t> left_storage,right_storage;
        auto left=source(row.original_left,row.type,row.shape,row.left_strides,left_storage),right=source(row.original_right,row.type,row.shape,row.right_strides,right_storage);
        const auto before_left=left_storage,before_right=right_storage;
        auto result=join_segmentations(left,right,{row.type,row.type,row.type,row.mapping});
        check(result.joined.data==row.joined);check(result.joined.shape==row.shape);
        check(result.joined.type.bytes==row.joined_type.bytes && result.joined.type.is_signed==row.joined_type.is_signed);
        check(result.has_mapping==row.mapping);
        if(row.mapping){check(result.first_map().keys==result.second_map().keys);check(result.keys.data==row.keys);check(result.first_values.data==row.first);check(result.second_values.data==row.second);}
        else check(result.keys.data.empty());
        check(left_storage==before_left && right_storage==before_right);
    }
    const JoinType u8{1,false},u16{2,false},i8{1,true},i16{2,true},u64{8,false};
    // Literal ORIGINAL factor201 forces a wrapped ordering unlike an abstract
    // pair join. Missing backward-map entries remain zero after divmod.
    const std::uint8_t original_left[]{0,7,42},original_right[]{0,200,0};
    ComponentsView lv{original_left,PixelType::uint8,false,false,{3},{1}},rv{original_right,PixelType::uint8,false,false,{3},{1}};
    auto wrapped=join_segmentations(lv,rv,{u8,u8,u8,true});
    check(wrapped.joined.data==std::vector<std::uint8_t>({0,2,1}));
    check(wrapped.keys.data==std::vector<std::uint8_t>({0,1,2}));
    check(wrapped.first_values.data==std::vector<std::uint8_t>({0,0,7}));
    check(wrapped.second_values.data==std::vector<std::uint8_t>({0,0,200}));
    auto no_maps=join_segmentations(lv,rv,{u8,u8,u8});check(!no_maps.has_mapping && no_maps.keys.data.empty());check(no_maps.joined.data==wrapped.joined.data);
    const std::uint16_t wide_right[]{0,200,0};rv={wide_right,PixelType::uint16,false,false,{3},{2}};
    auto promoted=join_segmentations(lv,rv,{u8,u16,u16,true});
    const std::uint16_t promoted_expected[]{0,2,1};
    check(promoted.joined.type.bytes==2 && !promoted.joined.type.is_signed);
    check(promoted.joined.data.size()==sizeof(promoted_expected) && std::memcmp(promoted.joined.data.data(),promoted_expected,sizeof(promoted_expected))==0);
    check(promoted.first_values.type.bytes==1 && promoted.second_values.type.bytes==2);
    const std::vector<std::uint8_t> a{2,3,4};const std::vector<std::uint16_t> b{1,2,3};
    std::vector<std::uint16_t> out(3);join_encode(a.data(),u8,b.data(),u16,3,200,u8,u16,out.data());
    check(out==std::vector<std::uint16_t>({145,90,35}));
    std::vector<std::int16_t> signed_out(3);join_encode(a.data(),u8,b.data(),u16,3,100,i8,i16,signed_out.data());
    check(signed_out==std::vector<std::int16_t>({-55,46,-109}));
    const std::size_t count=300019;std::vector<std::uint16_t> left(count),right(count),encoded(count);
    for(std::size_t i=0;i<count;++i){left[i]=std::uint16_t(i%97);right[i]=std::uint16_t(i%13);}
    const auto before_left=left,before_right=right;const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();
    ComponentsView whole_left{left.data(),PixelType::uint16,false,false,{count},{2}},whole_right{right.data(),PixelType::uint16,false,false,{count},{2}};
    auto whole=join_segmentations(whole_left,whole_right,{u16,u16,u16,true});
    check(whole.has_mapping && whole.first_map().keys==whole.second_map().keys);
    check(whole.joined.type.bytes==2 && whole.joined.shape==std::vector<std::size_t>{count});
    for(std::size_t i=0;i<count;++i){std::uint16_t label;std::memcpy(&label,whole.joined.data.data()+2*i,2);check(label==13*left[i]+right[i]);}
    for(std::size_t i=0;i<whole.keys.data.size()/2;++i){std::uint16_t key,first,second;std::memcpy(&key,whole.keys.data.data()+2*i,2);std::memcpy(&first,whole.first_values.data.data()+2*i,2);std::memcpy(&second,whole.second_values.data.data()+2*i,2);check(key==i && first==i/13 && second==i%13);}
    join_encode(left.data(),u16,right.data(),u16,count,200,u16,u16,encoded.data());
    std::vector<std::uint64_t> words(count);
    for(std::size_t i=0;i<count;++i){check(encoded[i]==200*left[i]+right[i]);words[i]=encoded[i];}
    const auto unique=consumer_sorted_unique(words.data(),words.size());
    std::vector<std::uint64_t> q(unique.size()),r(unique.size());
    join_divmod(unique.data(),u64,unique.size(),200,u64,q.data(),r.data());
    for(std::size_t i=0;i<unique.size();++i){check(q[i]<97);check(r[i]<13);check(q[i]*200+r[i]==unique[i]);}
    check(left==before_left && right==before_right);check(get_num_threads()==workers);
    check(std::fetestexcept(FE_ALL_EXCEPT)==flags);check(std::fegetround()==rounding);
    bool refused=false;try{join_divmod(unique.data(),u64,unique.size(),0,u64,q.data(),r.data());}catch(const std::invalid_argument&){refused=true;}
    check(refused);
    const std::uint8_t overflow_right[]{0,255,0};rv={overflow_right,PixelType::uint8,false,false,{3},{1}};
    refused=false;try{join_segmentations(lv,rv,{u8,u8,u8,true});}catch(const std::invalid_argument&){refused=true;}check(refused);
    refused=false;try{join_segmentations(lv,rv,{{8,false,true},u8,u8});}catch(const std::invalid_argument&){refused=true;}check(refused);
    return checks;
}
}
