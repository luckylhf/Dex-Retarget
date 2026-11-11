import numpy as np
from dex_frax.v8.robot import Robot
from dex_frax.v8.self_collision import CollisionModel
from dex_frax.local_geometry import project_root_height


def scene():
    r=Robot();row=np.r_[0.,0.,1.,np.zeros(29)];r.set(row,np.eye(3))
    floor=min(float((g['hull']@r.rot(g['foot'])[2]+r.point(g['foot'])[2]).min()) for g in r.sides.values())
    row[2]-=floor+.006;r.set(row,np.eye(3));n=9;rows=np.tile(row,(n,1))
    t=dict(root_r=np.tile(np.eye(3),(n,1,1)),body_contacts=[],sides={})
    for side,g in r.sides.items():
        pivot=g['sole'];anchor=r.point(g['foot'],pivot).copy();anchor[2]=0.
        t['sides'][side]=dict(contact=np.ones(n),pivots=np.tile(pivot,(n,1)),anchors=np.tile(anchor,(n,1)))
    return r,t,rows


def test_vertical_projection_fixes_ground_and_keeps_all_self_distances_and_hinges():
    r,t,rows=scene();out,detail=project_root_height(r,t,rows,0,len(rows))
    np.testing.assert_array_equal(out[:,3:],rows[:,3:]);np.testing.assert_array_equal(out[:,:2],rows[:,:2])
    assert .0059<detail['maximum_root_lift_m']<.0062
    model=CollisionModel(r,{'coverage':'full'})
    model.update(rows[0],np.eye(3));before=model.distances()
    model.update(out[0],np.eye(3));after=model.distances()
    np.testing.assert_allclose(after,before,atol=1e-9)
    for g in r.sides.values():
        height=float((g['hull']@r.rot(g['foot'])[2]+r.point(g['foot'])[2]).min())
        assert -.001<=height<=.001


def test_vertical_projection_preserves_joins_and_existing_body_support_corridor():
    r,t,rows=scene();t['body_contacts']=[dict(mode=np.ones(len(rows),int))]
    out,detail=project_root_height(r,t,rows,1,len(rows)-1)
    assert detail['maximum_root_lift_m']<=.000500001
    np.testing.assert_array_equal(out[:3],rows[:3]);np.testing.assert_array_equal(out[-3:],rows[-3:])
