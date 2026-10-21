"""Source verification must detect edits and survive ordinary checkouts."""
import pytest
from benchmarks.provenance import source_fingerprint

FILES={'CMakeLists.txt':'set(CMAKE_BUILD_TYPE Release)\n',
       'pyproject.toml':'[project]\nname="example"\n',
       'src/example.cpp':'int value() { return 1; }\n',
       'include/skimagexx/example.hpp':'int value();\n',
       'python/skimagexx/__init__.py':'VALUE = 1\n'}


def make_tree(root,newline=b'\n'):
    for name,text in FILES.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(text.encode().replace(b'\n',newline))


def test_fingerprint_survives_workspace_and_line_ending_changes(tmp_path):
    first,second=tmp_path/'one',tmp_path/'another-workspace'
    make_tree(first);make_tree(second,b'\r\n')
    assert source_fingerprint(first)==source_fingerprint(second)


@pytest.mark.parametrize('name',FILES)
def test_fingerprint_detects_source_and_build_changes(tmp_path,name):
    make_tree(tmp_path)
    before=source_fingerprint(tmp_path)
    path=tmp_path/name;path.write_bytes(path.read_bytes()+b'changed\n')
    assert source_fingerprint(tmp_path)!=before
