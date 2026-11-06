"""Science-free installed-family wiring checks; no SDK compilation or native calls."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_coords_scaled_sdk_default_and_focused_route():
    text=(ROOT/'tests/test_component_consumers_native.cpp').read_text()
    main=text[text.index('int main('):]
    route=main.index('if(argc==1||(argc==2&&std::string(argv[1])=="--regionprops-coords-scaled-only"))')
    call=main.index('regionprops_coords_scaled(workers);')
    focused_exit=main.index('if(argc==2){std::cout<<"installed coords_scaled controls=')
    legacy_route=main.index('if(argc==2&&std::string(argv[1])=="--regionprops-strided-only")')
    assert route<call<focused_exit<legacy_route
    assert 'for(int workers:{1,4})' in main[route:focused_exit]
def test_coords_scaled_reuses_installed_component_family():
    cmake=(ROOT/'tests/cmake_consumer/CMakeLists.txt').read_text()
    loop=cmake[cmake.index('foreach(_family IN ITEMS components component_consumers'):]
    assert 'add_executable(${_family}_consumer ../test_${_family}_native.cpp)' in loop
    assert 'add_test(NAME installed_${_family}_consumer COMMAND ${_family}_consumer ${_family_test_args})' in loop
    assert 'coords_scaled_consumer' not in cmake
