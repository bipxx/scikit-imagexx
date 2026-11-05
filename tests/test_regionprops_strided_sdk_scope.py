"""Pure source guard: the existing SDK family must execute new stride controls."""
from pathlib import Path


def test_existing_sdk_default_and_state_routes_execute_strided_controls():
    source = (Path(__file__).resolve().with_name('test_component_consumers_native.cpp')).read_text()
    main = source[source.index('int main('):]
    selected = 'if(argc==1||std::string(argv[1])=="--regionprops-state-only")'
    start = main.index(selected)
    end = main.index('})regionprops_scalars(workers);')
    assert start < end
    assert '})regionprops_strided(workers);' in main[start:end]
    assert main.count('})regionprops_strided(workers);') == 2
    assert '--regionprops-strided-only' in main[:start]
