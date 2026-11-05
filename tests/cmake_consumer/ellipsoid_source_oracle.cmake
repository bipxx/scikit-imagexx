# Nineteen exact saved source returns, both retained profiles have identical words.
set(SKIMAGEXX_ELLIPSOID_SOURCE_ORACLE "${CMAKE_CURRENT_LIST_DIR}/ellipsoid_source_oracle.bin")
if(NOT EXISTS "${SKIMAGEXX_ELLIPSOID_SOURCE_ORACLE}")
    message(FATAL_ERROR "The ellipsoid saved-source SDK oracle is missing")
endif()
file(SHA256 "${SKIMAGEXX_ELLIPSOID_SOURCE_ORACLE}" _ellipsoid_oracle_sha256)
if(NOT _ellipsoid_oracle_sha256 STREQUAL "8c9967242b2bb5ddfb66eeb551ad8a9367c0fc446a12efff16a98d48a39290ec")
    message(FATAL_ERROR "The ellipsoid saved-source SDK oracle differs")
endif()
