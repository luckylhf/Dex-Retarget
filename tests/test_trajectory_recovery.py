import numpy as np
from dex_frax.trajectory_recovery import basis


def test_continuous_seed_keeps_fixed_joins_and_releases_only_selected_cut():
    fixed=basis(101,True,True)
    np.testing.assert_array_equal(fixed[:2],0.)
    np.testing.assert_array_equal(fixed[-2:],0.)
    assert np.all(fixed>=0.) and np.max(fixed.sum(axis=1))<=1.+1e-12
    released=basis(101,True,False)
    np.testing.assert_array_equal(released[:2],0.)
    np.testing.assert_allclose(released[-1].sum(),1.)
    assert np.max(abs(np.diff(released,axis=0)))<.1
