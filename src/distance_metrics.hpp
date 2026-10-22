#pragma once
#include "distance_detail.hpp"
#include "distance_svml.hpp"
#include <limits>

namespace skimagexx::detail {
constexpr double distance_pi=3.141592653589793238462643383279502884;
// R is physically double in both implementations. The float implementation
// rounds each ordinary operation to float, while Wide retains promoted stages.
template<class T,bool Svml=false> struct ScalarDistanceMath {
    using R=double;
    using Wide=ScalarDistanceMath<double,Svml>;
    static R round(R x){return T(x);}
    static R c(double x){return T(x);}
    static R add(R a,R b){return T(T(a)+T(b));}
    static R sub(R a,R b){return T(T(a)-T(b));}
    static R mul(R a,R b){return T(T(a)*T(b));}
    static R div(R a,R b){return T(T(a)/T(b));}
    static R square(R a){return mul(a,a);}
    static R sqrt(R x){return std::sqrt(T(x));}
    static R power(R x,int n){
        if constexpr(Svml)if(distance_math_callbacks && distance_math_callbacks->power)
            return T(distance_math_callbacks->power(T(x),T(n),std::is_same_v<T,float>,distance_math_callbacks->context));
        return std::pow(T(x),T(n));
    }
    static R hypot(R a,R b){
        if constexpr(Svml)if(distance_math_callbacks && distance_math_callbacks->hypot)
            return T(distance_math_callbacks->hypot(T(a),T(b),std::is_same_v<T,float>,distance_math_callbacks->context));
        return std::hypot(T(a),T(b));
    }
    static R angle(R a,R b){
        const auto h=[](T x,T y){
            if constexpr(Svml && std::is_same_v<T,double>)return svml_atan2(y,x);
            else return std::atan2(y,x);
        }(T(a),T(b));
        return T(double(h)+(h<0?2*distance_pi:0));
    }
    static R sin(R x){return std::sin(T(x));}
    static R cos(R x){return std::cos(T(x));}
    static R exp(R x){return std::exp(T(x));}
    static R abs(R x){return std::abs(T(x));}
    static R degrees(R x){return mul(x,c(T(180)/T(distance_pi)));}
    static R nonnegative(R x){return x<=0?0:x;}
    static bool lt(R a,R b){return a<b;}
    static bool gt(R a,R b){return a>b;}
    static bool eq(R a,R b){return a==b;}
    static bool ge(R a,R b){return a>=b;}
    static bool le(R a,R b){return a<=b;}
    static bool both(bool a,bool b){return a&&b;}
    static R choose(R no,R yes,bool mask){return mask?yes:no;}
    static R hue_squared(R a,R b,R x,R y){
        return round(2*(std::hypot(a,b)*std::hypot(x,y)-(a*x+b*y)));
    }
};

// M controls the lightness/accumulation dtype; C controls chroma/hue.
// NumPy 1.x promotes c7 + 25**7 to double for float32 arrays. NumPy 2
// keeps the weak Python integer in float32. In-place sums still round to M.
template<class M,class C,class W> typename M::R ciede2000_distance(
    typename M::R (&a)[3],typename M::R (&b)[3],const DistanceParameters& p,
    typename M::R initial_c1,typename M::R initial_c2) {
    const auto one=C::c(1),two=C::c(2),zero=C::c(0);
    const auto L1=a[0],L2=b[0];
    auto Cbar=M::mul(M::c(.5),M::add(initial_c1,initial_c2));
    auto c7=M::power(Cbar,7);
    const auto G=C::mul(C::c(.5),C::sub(one,C::sqrt(C::div(c7,C::add(c7,C::c(6103515625.))))));
    const auto scale=C::add(one,G),a1=C::mul(a[1],scale),a2=C::mul(b[1],scale);
    auto C1=C::hypot(a1,a[2]),C2=C::hypot(a2,b[2]);
    const auto h1=C::angle(a1,a[2]),h2=C::angle(a2,b[2]);
    const auto Lbar=M::mul(M::c(.5),M::add(L1,L2)),tmp=M::square(M::sub(Lbar,M::c(50)));
    const auto SL=M::add(one,M::div(M::mul(M::c(.015),tmp),M::sqrt(M::add(M::c(20),tmp))));
    const auto Lterm=M::div(M::sub(L2,L1),M::mul(M::c(p.kL),SL));
    Cbar=C::mul(C::c(.5),C::add(C1,C2));
    const auto SC=C::add(one,C::mul(C::c(.045),Cbar));
    const auto Cterm=C::div(C::sub(C2,C1),C::mul(C::c(p.kC),SC));
    const auto diff=C::sub(h2,h1),sum=C::add(h1,h2),CC=C::mul(C1,C2);
    auto dH=C::choose(diff,C::sub(diff,C::c(2*distance_pi)),C::gt(diff,C::c(distance_pi)));
    dH=C::choose(dH,C::add(dH,C::c(2*distance_pi)),C::lt(diff,C::c(-distance_pi)));
    dH=C::choose(dH,zero,C::eq(CC,zero));
    const auto dHterm=C::mul(C::mul(two,C::sqrt(CC)),C::sin(C::div(dH,two)));
    const auto mask=C::both(C::gt(C::abs(diff),C::c(distance_pi)),C::gt(CC,zero));
    auto Hbar=C::choose(sum,C::add(sum,C::c(2*distance_pi)),C::both(mask,C::lt(sum,C::c(2*distance_pi))));
    Hbar=C::choose(Hbar,C::sub(Hbar,C::c(2*distance_pi)),C::both(mask,C::ge(sum,C::c(2*distance_pi))));
    Hbar=C::choose(Hbar,C::mul(Hbar,two),C::eq(CC,zero));
    Hbar=C::mul(Hbar,C::c(.5));
    auto T=W::sub(W::c(1),W::mul(W::c(.17),W::cos(W::sub(Hbar,W::c(30*distance_pi/180)))));
    T=W::add(T,C::mul(C::c(.24),C::cos(C::mul(two,Hbar))));
    T=W::add(T,W::mul(W::c(.32),W::cos(W::add(C::mul(C::c(3),Hbar),W::c(6*distance_pi/180)))));
    T=W::sub(T,W::mul(W::c(.20),W::cos(W::sub(C::mul(C::c(4),Hbar),W::c(63*distance_pi/180)))));
    const auto SH=W::add(W::c(1),W::mul(C::mul(C::c(.015),Cbar),T));
    const auto Hterm=W::div(dHterm,W::mul(W::c(p.kH),SH));
    c7=C::power(Cbar,7);
    const auto Rc=C::mul(two,C::sqrt(C::div(c7,C::add(c7,C::c(6103515625.)))));
    const auto q=C::div(C::sub(C::degrees(Hbar),C::c(275)),C::c(25));
    const auto dtheta=W::mul(W::c(30*distance_pi/180),C::exp(C::sub(zero,C::square(q))));
    const auto rotation=W::mul(W::mul(W::mul(W::sub(W::c(0),W::sin(W::mul(W::c(2),dtheta))),Rc),Cterm),Hterm);
    auto dE2=M::square(Lterm);
    dE2=M::round(C::add(dE2,C::square(Cterm)));
    dE2=M::round(W::add(dE2,W::square(Hterm)));
    dE2=M::round(W::add(dE2,rotation));
    return M::sqrt(M::nonnegative(dE2));
}

template<class M,class W> typename M::R weighted_distance(typename M::R (&a)[3],typename M::R (&b)[3],
    DistanceMetric metric,const DistanceParameters& p,bool promoted_cmc,bool legacy_ciede2000) {
    using R=typename M::R;
    const auto one=M::c(1);
    const auto L1=a[0],L2=b[0];
    auto C1=M::hypot(a[1],a[2]),C2=M::hypot(b[1],b[2]);
    if(metric==DistanceMetric::cie94 || metric==DistanceMetric::cmc) {
        const auto dL=M::sub(L1,L2),dC=M::sub(C1,C2);
        const auto dH2=M::hue_squared(a[1],a[2],b[1],b[2]);
        if(metric==DistanceMetric::cie94) {
            const auto SC=M::add(one,M::mul(M::c(p.k1),C1)),SH=M::add(one,M::mul(M::c(p.k2),C1));
            auto dE2=M::square(M::div(dL,M::c(p.kL)));
            dE2=M::add(dE2,M::square(M::div(dC,M::mul(M::c(p.kC),SC))));
            dE2=M::add(dE2,M::div(dH2,M::square(M::mul(M::c(p.kH),SH))));
            return M::sqrt(M::nonnegative(dE2));
        }
        const auto hue=M::angle(a[1],a[2]);
        const auto degrees=M::degrees(hue);
        const auto in_sector=M::both(M::ge(degrees,M::c(164)),M::le(degrees,M::c(345)));
        const auto t0=W::add(W::c(.36),W::mul(W::c(.4),W::abs(W::cos(W::add(hue,W::c(35*distance_pi/180))))));
        const auto t1=W::add(W::c(.56),W::mul(W::c(.2),W::abs(W::cos(W::add(hue,W::c(168*distance_pi/180))))));
        const auto factor=W::choose(t0,t1,in_sector);
        const auto c4=M::power(C1,4),F=M::sqrt(M::div(c4,M::add(c4,M::c(1900))));
        const auto sl_curve=M::div(M::mul(M::c(.040975),L1),M::add(one,M::mul(M::c(.01765),L1)));
        const auto SL=M::choose(sl_curve,M::c(.511),M::lt(L1,M::c(16)));
        const auto SC=M::add(M::c(.638),M::div(M::mul(M::c(.0638),C1),M::add(one,M::mul(M::c(.0131),C1))));
        const auto SH=W::mul(SC,W::sub(W::add(W::mul(F,factor),W::c(1)),F));
        const auto Cterm=M::square(M::div(dC,M::mul(M::c(p.kC),SC)));
        const auto Hterm=W::div(dH2,W::square(SH));
        auto dE2=M::square(M::div(dL,M::mul(M::c(p.kL),SL)));
        dE2=M::add(dE2,Cterm);
        if(promoted_cmc) {
            using D=typename M::Wide;
            return D::sqrt(D::nonnegative(D::add(dE2,Hterm)));
        }
        dE2=M::round(W::add(dE2,Hterm));
        return M::sqrt(M::nonnegative(dE2));
    }
    if(legacy_ciede2000)
        return ciede2000_distance<M,typename M::Wide,typename M::Wide>(a,b,p,C1,C2);
    return ciede2000_distance<M,M,W>(a,b,p,C1,C2);
}
template<class M> typename M::R distance_metric(typename M::R (&a)[3],typename M::R (&b)[3],
    DistanceMetric metric,const DistanceParameters& p,bool wide_trig,bool promoted_cmc,bool legacy_ciede2000=false) {
    return wide_trig ? weighted_distance<M,typename M::Wide>(a,b,metric,p,promoted_cmc,legacy_ciede2000) :
                       weighted_distance<M,M>(a,b,metric,p,promoted_cmc,legacy_ciede2000);
}
}
