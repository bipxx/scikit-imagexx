// Signed-zero tie handling follows SciPy's rank_filter_1d.cpp mediator.
// Copyright (c) 2011 ashelly.myopenid.com; modified in 2024 by Gideon Genadi Kogan.
// MIT license: see third_party/scipy-rank1d-LICENSE.txt.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

namespace skimagexx { namespace detail {
// A separate rank node, with negative max-heap and positive min-heap indices,
// retains upstream's insertion history and strict-comparison tie ordering.
template<class T> class MedianTies {
    std::vector<T> storage;
    std::vector<std::int32_t> indices;
    T* values;
    std::int32_t* positions;
    std::int32_t* heap;
    std::ptrdiff_t below,above;
    const std::size_t count;
    std::size_t next=0;
    auto& at(std::ptrdiff_t p) {return heap[p];}
    bool less(std::ptrdiff_t a,std::ptrdiff_t b) {return values[at(a)]<values[at(b)];}
    bool exchange_if_less(std::ptrdiff_t a,std::ptrdiff_t b) {
        if(!less(a,b))return false;
        std::swap(at(a),at(b));positions[at(a)]=std::int32_t(a);positions[at(b)]=std::int32_t(b);
        return true;
    }
    void down_min(std::ptrdiff_t p) {
        for(p*=2;p<=above;p*=2) {
            if(p<above && less(p+1,p))++p;
            if(!exchange_if_less(p,p/2))break;
        }
    }
    void down_max(std::ptrdiff_t p) {
        for(p*=2;p>=-below;p*=2) {
            if(p>-below && less(p,p-1))--p;
            if(!exchange_if_less(p/2,p))break;
        }
    }
    bool up_min(std::ptrdiff_t p) {
        while(p>0 && exchange_if_less(p,p/2))p/=2;
        return p==0;
    }
    bool up_max(std::ptrdiff_t p) {
        while(p<0 && exchange_if_less(p/2,p))p/=2;
        return p==0;
    }
public:
    MedianTies(std::size_t length,std::size_t selected):storage(length),indices(2*length),
        values(storage.data()),positions(indices.data()),heap(positions+length+selected),
        below(std::ptrdiff_t(selected)),above(std::ptrdiff_t(length-selected-1)),count(length) {
        for(std::size_t i=0;i<count;++i){positions[i]=std::int32_t(std::ptrdiff_t(i)-below);at(positions[i])=std::int32_t(i);}
    }
    MedianTies(const MedianTies&)=delete;
    MedianTies& operator=(const MedianTies&)=delete;
    void insert(T value) {
        const auto p=positions[next];const auto old=values[next];values[next]=value;
        if(++next==count)next=0;
        if(p>0) {
            if(value>old){down_min(p);return;}
            if(up_min(p) && exchange_if_less(0,-1))down_max(-1);
        } else if(p<0) {
            if(value<old){down_max(p);return;}
            if(up_max(p) && exchange_if_less(1,0))down_min(1);
        } else {
            if(up_max(-1))down_max(-1);
            if(up_min(1))down_min(1);
        }
    }
    T value() {return values[at(0)];}
};

template<class T,class W,class Store>
void median_tie_line(const T* input,std::size_t n,std::size_t window,std::size_t rank,
                     GaussianBorder border,W constant,Store store) {
    MedianTies<W> state(window,rank);
    const auto tail=(window-1)/2;
    if(border==GaussianBorder::reflect) {
        for(auto i=window-tail;i-->0;)state.insert(W(input[i]));
    } else if(border==GaussianBorder::mirror) {
        for(auto i=window-tail;i>0;--i)state.insert(W(input[i]));
    } else if(border==GaussianBorder::wrap) {
        const auto extra=window%2==0 ? 2 : 0;
        for(auto i=n-tail-extra;i<n;++i)state.insert(W(input[i]));
    } else {
        const W value=border==GaussianBorder::constant ? constant : W(input[0]);
        for(std::size_t i=0;i<window-tail;++i)state.insert(value);
    }
    for(std::size_t i=0;i<tail;++i)state.insert(W(input[i]));
    for(auto i=tail;i<n;++i){state.insert(W(input[i]));store(i-tail,state.value());}
    for(std::size_t i=0;i<tail;++i) {
        W value=constant;
        if(border==GaussianBorder::nearest)value=W(input[n-1]);
        else if(border==GaussianBorder::reflect)value=W(input[n-1-i]);
        else if(border==GaussianBorder::mirror)value=W(input[n-2-i]);
        else if(border==GaussianBorder::wrap)value=W(input[i]);
        state.insert(value);store(n-tail+i,state.value());
    }
}
}}
