import numpy as np
from scipy.spatial.transform import Rotation
from dex_frax.shoulder_reference import continuous_coordinates,prepare
from dex_frax.v8.robot import Robot


def test_future_limit_resolves_equivalent_shoulder_coordinates_continuously():
    target=np.deg2rad(np.c_[np.linspace(140,175,70),np.full(70,-60.),np.full(70,-20.)])
    rotation=Rotation.from_euler('YXZ',target).as_matrix()
    lo=np.deg2rad([-163.,-191.,-163.]);hi=np.deg2rad([163.,11.,163.]);step=np.full(3,.12)
    ref,report=continuous_coordinates(rotation,lo,hi,step)
    assert np.all(ref>=lo-2e-6) and np.all(ref<=hi+2e-6)
    assert np.max(abs(np.diff(ref,axis=0))-step)<2e-6
    # Start in the representation that remains feasible at the future endpoint.
    assert ref[0,0]<0.
    actual=Rotation.from_euler('YXZ',ref).as_matrix()
    assert np.max(Rotation.from_matrix(actual.transpose(0,2,1)@rotation).magnitude())<.002


def test_preparation_preserves_caller_and_uses_final_protected_body():
    r=Robot();n=20;rows=np.tile(np.r_[0.,0.,1.,np.zeros(29)],(n,1))
    rows[:,15]=np.linspace(-.2,.2,n);root=np.tile(np.eye(3),(n,1,1));sides={s:dict(upper=[],elbow_axis=[]) for s in r.sides}
    for i,row in enumerate(rows):
        r.set(row,root[i])
        for side,g in r.sides.items():
            direction=r.point(g['elbow'])-r.point(g['shoulder']);direction/=np.linalg.norm(direction)
            sides[side]['upper'].append(direction);sides[side]['elbow_axis'].append(-r.rot(g['shoulder'])[:,1])
    sides={s:{k:np.asarray(v) for k,v in values.items()} for s,values in sides.items()}
    saved=r.d.qpos.copy();target=dict(priority_pose=rows,root_r=root,sides=sides,fps=50.)
    out,report=prepare(r,target,{'numerical_joint_speed_deg_s':720.})
    np.testing.assert_array_equal(r.d.qpos,saved)
    assert out['shoulder_reference'].shape==(n,6)
    assert np.isfinite(out['shoulder_reference']).all()
    assert report['used'] and 'shoulder_reference' not in target
    # A correct source-relative arm follows the moving torso with near-zero
    # local shoulders, rather than counterrotating the protected waist.
    assert np.max(abs(out['shoulder_reference']))<.002
