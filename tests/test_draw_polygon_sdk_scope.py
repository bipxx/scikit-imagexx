"""Installed drawing family wiring only; no scientific imports or execution."""
from pathlib import Path

ROOT=Path(__file__).parents[1]


def test_existing_drawing_sdk_default_executes_polygon_controls():
    text=(ROOT/'tests/cmake_consumer/draw_coordinates_consumer.cpp').read_text()
    main=text[text.index('int main('):]
    assert '#include <skimagexx/draw_polygon.hpp>' in text
    assert 'for(int workers:{1,4}){set_num_threads(workers);' in main
    assert 'if(argc==1||polygon_only)polygon(workers);' in main
    assert 'if(!polygon_only){' in main and 'stages(workers)' in main
    assert 'set_num_threads(1);check(get_num_threads()==1);' in main
    assert '--polygon-only' in main
    assert 'draw_polygon_prepare(' in text and 'draw_polygon_coords_into(' in text and 'draw_polygon_mask_into(' in text
