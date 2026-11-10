import numpy as np
from dex_frax.v8.robot import Robot
from dex_frax.palm_constraints import SourcePalms


def test_palm_constraint_matches_jacobian_and_existing_envelope():
    r=Robot();g=r.sides['Left'];x=np.r_[0.,0.,1.,np.zeros(29)];ids=np.asarray(g['arm']);x[ids[3]]=-.4
    r.set(x,np.eye(3));R=r.rot(g['tcp']).copy();t={'root_r':np.eye(3)[None],'sides':{'Left':{'hand_forward':(-R[:,2])[None],'hand_width':R[:,0][None]}}}
    model=SourcePalms(r,t,x[None],0,1,['Left'],ids)
    np.testing.assert_allclose(model.limits,np.deg2rad(25.))
    row=x.copy();row[ids]+=np.array([.2,.1,.1,-.1,.2,.1,.1]);C,h,_=model.evaluate(row,0);numeric=[]
    for j in ids:
        d=np.zeros(32);d[j]=1e-6;numeric.append(-(model.evaluate(row+d,0)[1]-model.evaluate(row-d,0)[1])/(2e-6))
    np.testing.assert_allclose(C,np.array(numeric).T,atol=1e-6)
    # An already inaccurate reference is protected at its existing error.
    t['sides']['Left']['hand_forward']*= -1
    bad=SourcePalms(r,t,x[None],0,1,['Left'],ids)
    assert bad.limits[0,0,0]>np.deg2rad(179.9)


def test_interior_margin_tightens_search_but_does_not_change_acceptance():
    r=Robot();g=r.sides['Left'];ids=np.asarray(g['arm']);x=np.r_[0.,0.,1.,np.zeros(29)]
    r.set(x,np.eye(3));R=r.rot(g['tcp']).copy();t={'root_r':np.tile(np.eye(3),(2,1,1)),'sides':{'Left':{'hand_forward':np.tile(-R[:,2],(2,1)),'hand_width':np.tile(R[:,0],(2,1))}}}
    plain=SourcePalms(r,t,np.tile(x,(2,1)),0,2,['Left'],ids)
    interior=SourcePalms(r,t,np.tile(x,(2,1)),0,2,['Left'],ids,interior_deg=.1,fixed_frames=[0])
    np.testing.assert_allclose(plain.evaluate(x,0)[1],interior.evaluate(x,0)[1])
    assert np.all(interior.evaluate(x,1)[1]<plain.evaluate(x,1)[1])
    assert plain.evaluate(x,1)[2]==interior.evaluate(x,1)[2]
    np.testing.assert_array_equal(plain.limits,interior.limits)
