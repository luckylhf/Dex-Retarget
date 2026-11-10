import pytest
from dex_frax.dense_floor_refine import refine


def test_dense_floor_refuses_unbounded_or_too_short_windows():
    for size in (4,32,251):
        with pytest.raises(ValueError,match='dense_floor_window_budget'):
            refine(None,None,None,None,0,size,float('inf'))


def test_fixed_passing_join_does_not_have_to_meet_new_search_margin():
    from dex_frax.dense_floor_refine import floor_target
    assert floor_target(-.000964,True)<-.000964
    assert floor_target(-.000964,False)==-.00095
    assert floor_target(-.002,True)==-.001
