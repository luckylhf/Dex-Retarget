import numpy as np
from dex_frax.recovery_seed import project


def test_discontinuous_seed_becomes_speed_feasible_without_moving_joins():
    q=np.zeros((20,2));q[2:18,0]=1.5;q[8:12,1]=-1.
    joins=np.zeros_like(q);frozen=np.zeros_like(q,dtype=bool);frozen[:2]=True;frozen[-2:]=True
    lower=np.full_like(q,-1.6);upper=np.full_like(q,1.6);step=np.array([.1,.15])
    out,report=project(q,lower,upper,step,frozen,joins,np.zeros(2),np.zeros(2))
    assert report['executed']
    np.testing.assert_array_equal(out[frozen],joins[frozen])
    assert np.max(abs(np.diff(out,axis=0))-step)<1e-7
    assert np.all(out>=lower-1e-7) and np.all(out<=upper+1e-7)
    assert out[10,0]>.2


def test_movable_seam_still_respects_outside_velocity():
    q=np.ones((6,2));step=np.array([.1,.2]);fixed=np.zeros_like(q,dtype=bool)
    out,report=project(q,np.full_like(q,-2.),np.full_like(q,2.),step,fixed,np.zeros_like(q),np.zeros(2),np.zeros(2))
    assert np.all(abs(out[0])<=step+1e-7) and np.all(abs(out[-1])<=step+1e-7)
    assert np.all(abs(np.diff(out,axis=0))<=step+1e-7)


def test_feasible_seed_is_unchanged():
    q=np.zeros((5,2));out,report=project(q,-np.ones_like(q),np.ones_like(q),np.array([.1,.1]),np.zeros_like(q,dtype=bool),q)
    np.testing.assert_array_equal(out,q);assert not report['executed']
