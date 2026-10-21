#pragma once
namespace skimagexx::detail {
// NumPy float32 rational-log coefficients (BSD notice in third_party).
inline constexpr float threshold_log_p[]={0.0f,9.999999999999998702752e-01f,2.112677543073053063722e+00f,
    1.480000633576506585156e+00f,3.808837741388407920751e-01f,2.589979117907922693523e-02f};
inline constexpr float threshold_log_q[]={1.0f,2.612677543073109236779e+00f,2.453006071784736363091e+00f,
    9.864942958519418960339e-01f,1.546476374983906719538e-01f,5.875095403124574342950e-03f};
inline constexpr float threshold_log_ln2=0.6931471805599453094f;
inline constexpr float threshold_log_sqrt_half=0.7071067811865475244f;
}
