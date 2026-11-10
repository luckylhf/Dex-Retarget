import numpy as np
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from dex_frax.retiming import apply,time_map


def test_local_slowdown_preserves_endpoints_and_caps_root_speed():
    r=Rotation.from_euler('z',np.array([0,.01,.8,.82,.83])[:,None]).as_matrix();p=np.zeros((5,3))
    idx,report=time_map(r,p,50.,7.8,7.8)
    assert report['changed'] and report['stretched_intervals']==[1]
    assert idx[0]==0. and idx[-1]==4. and np.all(np.diff(idx)>=0)
    t=dict(fps=50.,root_r=r,root_p=p,hips=p.copy(),sides={},scale=1.2,floor=.04,world_origin=np.array([.1,.2,.3]))
    out,clock,meta=apply(t,SimpleNamespace(sides={}),7.8,7.8)
    speed=Rotation.from_matrix(out['root_r'][:-1].transpose(0,2,1)@out['root_r'][1:]).magnitude()*50
    assert speed.max()<=7.8+1e-10
    np.testing.assert_allclose(out['root_r'][[0,-1]],r[[0,-1]],atol=1e-12)
    np.testing.assert_array_equal(out['world_origin'],t['world_origin'])
    assert out['scale']==1.2 and out['floor']==.04 and len(t['root_r'])==5
    np.testing.assert_allclose(clock,idx/50)


def test_no_slowdown_is_exact_identity():
    rr=np.tile(np.eye(3),(5,1,1));t=dict(fps=50.,root_r=rr,root_p=np.zeros((5,3)))
    out,clock,meta=apply(t,None,7.8,7.8)
    assert out is t and not meta['changed']
    np.testing.assert_array_equal(clock,np.arange(5)/50)


def test_contact_switches_keep_discrete_anchor_and_full_calibration():
    n=5;rr=Rotation.from_euler('z',np.array([0,.01,.8,.82,.83])[:,None]).as_matrix();eye=np.tile(np.eye(3),(n,1,1))
    anchors=np.zeros((n,3));anchors[1:3,0]=2.
    v=dict(contact=np.array([0.,1.,1.,0.,0.]),phase=np.array([0,1,1,0,0],np.int8),anchors=anchors,pivots=np.zeros((n,3)),
           foot_target_p=anchors.copy(),foot_target_r=eye,heel=np.zeros((n,3)),toe=np.ones((n,3)),
           upper=np.tile([1.,0.,0.],(n,1)),segments=[dict(start=1,end=3,label=1)],foot_calibration={'offset':.2})
    t=dict(fps=50.,root_r=rr,root_p=np.zeros((n,3)),sides={'Left':v},ground_calibration={'fixed':True})
    r=SimpleNamespace(sides={'Left':{'hull':np.zeros((4,3))}})
    out,time,_=apply(t,r,7.8,7.8);f=out['sides']['Left'];active=f['contact']>0
    assert set(f['contact'])=={0.,1.}
    np.testing.assert_array_equal(f['anchors'][active,0],2.)
    np.testing.assert_array_equal(f['foot_target_p'][active,0],2.)
    assert out['ground_calibration']==t['ground_calibration'] and f['foot_calibration']==v['foot_calibration']
    assert np.all(f['upper']==[1.,0.,0.])
    segment=f['segments'][0];assert np.array_equal(np.flatnonzero(active),np.arange(segment['start'],segment['end']))


def test_untouched_prefix_keeps_exact_numeric_targets():
    rr=Rotation.from_euler('z',np.array([0,.01,.02,.03,.8])[:,None]).as_matrix()
    direction=np.tile([1.,1e-12,0.],(5,1));p=np.arange(15,dtype=float).reshape(5,3)*1e-4
    t=dict(fps=50.,root_r=rr,root_p=p,hips=p.copy(),sides={},upper=direction)
    out,clock,_=apply(t,SimpleNamespace(sides={}),7.8,7.8)
    for key in ('root_r','root_p','hips','upper'):np.testing.assert_array_equal(out[key][:4],t[key][:4])
    np.testing.assert_array_equal(clock[:4],np.arange(4)/50)
