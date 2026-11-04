#include <skimagexx/line_aa.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_line_aa(py::module_& module) {
    module.def("draw_line_aa", [](py::ssize_t r0, py::ssize_t c0,
                                  py::ssize_t r1, py::ssize_t c1, int workers) -> py::object {
        if (workers < 1 || workers > 1024)
            throw py::value_error("drawing worker snapshot outside [1,1024]");
        skimagexx::detail::DrawLineAAGeometry geometry;
        if (!skimagexx::detail::draw_line_aa_prepare(r0, c0, r1, c1, geometry))
            return py::none();
        const auto count = skimagexx::detail::draw_line_aa_count(geometry, workers);
        // Audited C-int squared deltas bound the count well below Py_ssize_t.
        py::array_t<std::intptr_t> rows{py::ssize_t(count)}, columns{py::ssize_t(count)};
        py::array_t<double> weights{py::ssize_t(count)};
        const skimagexx::detail::DrawCoordinatePair output{
            rows.mutable_data(), columns.mutable_data(), 1, 1};
        auto* values = weights.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::detail::draw_line_aa_into(geometry, output, values, 1, workers);
        }
        return py::make_tuple(rows, columns, weights);
    }, py::arg("r0"), py::arg("c0"), py::arg("r1"), py::arg("c1"), py::arg("workers"));
}
