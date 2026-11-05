# Twenty-three unique immutable original returns; both retained profiles agree.
set(SKIMAGEXX_ELLIPSE_PERIMETER_FIXTURES "${CMAKE_CURRENT_LIST_DIR}/ellipse_perimeter_fixtures.bin")
if(NOT EXISTS "${SKIMAGEXX_ELLIPSE_PERIMETER_FIXTURES}")
  message(FATAL_ERROR "Ellipse perimeter saved-source fixtures missing")
endif()
file(SHA256 "${SKIMAGEXX_ELLIPSE_PERIMETER_FIXTURES}" _ellipse_perimeter_sha)
if(NOT _ellipse_perimeter_sha STREQUAL "079da2ae19560e01b5e098a6a12121046479392ced66d6572ff648b2448f5772")
  message(FATAL_ERROR "Ellipse perimeter saved-source fixtures differ")
endif()
