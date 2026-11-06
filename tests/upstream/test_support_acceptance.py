import pytest
from dex_frax.v8.support_acceptance import support_constraint_failed

@pytest.mark.parametrize('floor,error,failed',[
    (0.,.0001289497283,False), (-.0009,.0009,False),
    (-.001,.001,False),(-.001001,0.,True),(0.,.001001,True),
    (float('nan'),0.,True),(0.,float('inf'),True),
])
def test_final_support_acceptance(floor,error,failed):
    assert support_constraint_failed(floor,error) is failed
