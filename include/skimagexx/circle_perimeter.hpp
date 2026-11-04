#pragma once
#include <skimagexx/draw_coordinates.hpp>
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
enum class CirclePerimeterMethod { bresenham, andres };
struct CirclePerimeterRequest {
    std::intptr_t row=0,column=0,radius=0;
    CirclePerimeterMethod method=CirclePerimeterMethod::bresenham;
    bool clipped=false;
    std::intptr_t height=0,width=0;
};
// Established once by prepare_circle_perimeter. Preserve this value unchanged
// between count and fill; no Python objects/callbacks or external buffers exist.
struct CirclePerimeter {
    CirclePerimeterRequest request;
    std::size_t max_points=0;
};
// Refuses unknown methods or decision, coordinate-addition and address-size
// arithmetic outside the prepared partition. A refusal leaves prepared intact.
// Negative radii are empty; zero emits eight repeated center points. Shape
// dimensions are literal prepared integer bounds; nonpositive bounds clip all.
bool prepare_circle_perimeter(const CirclePerimeterRequest&,CirclePerimeter&) noexcept;
// Both methods preserve the original eight-reflection ordering and duplicate
// points. Clipping is stable and occurs after integer origin addition. The
// recurrence is dependent and serial; workers is the caller's global snapshot.
std::size_t circle_perimeter_count(const CirclePerimeter&,int workers) noexcept;
// Caller supplies fresh disjoint aligned intp storage of exact count capacity.
// Element steps may be positive or negative; address spans/owners are caller-
// validated once. Empty output permits null pointers. No allocation or FP work.
void circle_perimeter_fill(const CirclePerimeter&,DrawCoordinatePair,int workers) noexcept;
}
