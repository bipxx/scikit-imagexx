#include "distance_metrics.hpp"

// Diagnostic primitive only: exactly the existing scalar hue graph, with
// runtime operands. The production implementation and SDK remain untouched.
extern "C" __attribute__((visibility("default")))
double weighted_hue_graph_v1(double a, double b, double x, double y) {
    return skimagexx::detail::ScalarDistanceMath<double>::hue_squared(a, b, x, y);
}
