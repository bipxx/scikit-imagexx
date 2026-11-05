# Twenty-eight exact once-only original-success records; both profiles agree.
set(SKIMAGEXX_BEZIER_FIXTURE "${CMAKE_CURRENT_LIST_DIR}/../fixtures/bezier_curve_originals_v1.bin")
if(NOT EXISTS "${SKIMAGEXX_BEZIER_FIXTURE}")
  message(FATAL_ERROR "Bezier saved original fixtures missing")
endif()
file(SHA256 "${SKIMAGEXX_BEZIER_FIXTURE}" _bezier_fixture_sha)
if(NOT _bezier_fixture_sha STREQUAL "7bf92a2e8710833e4a065037576f755b2ca7278c86b5648615f87994893266fc")
  message(FATAL_ERROR "Bezier saved original fixtures differ")
endif()
