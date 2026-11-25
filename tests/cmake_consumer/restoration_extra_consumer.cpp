#include <skimagexx/restoration_extra.hpp>

#ifdef SKIMAGEXX_RESTORATION_EXTRA_PRIVATE
#include <skimagexx/image.hpp>
#include <pybind11/pybind11.h>

void bind_restoration_extra(pybind11::module_&);
PYBIND11_MODULE(_restoration_extra_native, module, pybind11::multiple_interpreters::not_supported()) {
    bind_restoration_extra(module);
    module.def("get_num_threads", &skimagexx::get_num_threads);
    module.def("set_num_threads", &skimagexx::set_num_threads);
    module.def("has_avx2", &skimagexx::has_avx2);
}
#else
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

template<class T> void check_kernels() {
    const skimagexx::RestorationShape shape{3, 4};
    const std::array<T, 12> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::array<std::complex<T>, 9> transfer{};
    std::array<T, 9> reg{};
    std::array<T, 12> output{};
    const T impulse = 1;
    skimagexx::restoration_transfer(&impulse, {1, 1}, shape, transfer.data(), 1);
    skimagexx::restoration_laplacian(shape, reg.data());
    if (transfer[8] != std::complex<T>(1) || std::abs(reg[8] - T(7)) > T(1e-6))
        throw std::runtime_error("unexpected transfer or Laplacian");
    if (!skimagexx::restoration_wiener(input.data(), shape, transfer.data(), reg.data(),
                                      T(0), false, output.data(), 1))
        throw std::runtime_error("identity Wiener declined");
    for (std::size_t i = 0; i < output.size(); ++i)
        if (std::abs(output[i] - input[i]) > T(1e-5))
            throw std::runtime_error("identity Wiener mismatch");

    skimagexx::restoration_interpolate(input.data(), shape, 2, output.data());
    if (output[0] != T(2.5) || output[5] != T(5) || output[11] != T(8.5))
        throw std::runtime_error("mirror interpolation mismatch");
    skimagexx::restoration_roll(input.data(), shape, {1, 2}, output.data(), false);
    if (output[0] != T(10) || output[4] != T(2) || output[11] != T(5))
        throw std::runtime_error("roll mismatch");
    skimagexx::restoration_roll(input.data(), shape, {0, 0}, output.data(), true);
    if (output[0] != T(10) || output[4] != T(6) || output[11] != T(16))
        throw std::runtime_error("roll accumulator mismatch");

    const std::complex<T> data(4, -2), weight(2, 0);
    const T precision = 2;
    const double a = 2, b = -4;
    std::complex<T> sample;
    skimagexx::restoration_gibbs_sample(&data, &weight, &precision, &a, &b, 1, &sample);
    if (sample != std::complex<T>(5, -4))
        throw std::runtime_error("Gibbs step mismatch");
}

int main() {
    check_kernels<float>();
    check_kernels<double>();
    std::cout << "restoration_extra kernels passed (float and double)\n";
}
#endif
