import numpy as np
import jax
import jax.numpy as jnp
from dex_frax.v8.robot import Robot
from dex_frax.v8.self_collision import CollisionModel
from dex_frax.refine_routing import collision_groups
from dex_frax.frax_arm_constraints import ArmGeometry


def test_initial_arm_geometry_matches_full_audit_pairs_and_preserves_caller():
    r=Robot();before=r.d.qpos.copy();constraint=ArmGeometry(object(),r,50.)
    np.testing.assert_array_equal(r.d.qpos,before)
    full=CollisionModel(r,{'coverage':'full'})
    expected={tuple(full.pairs[k]) for k in collision_groups(full)['collision']}
    assert set(map(tuple,constraint.geometry.pairs))==expected
    assert constraint.report['pair_count']==constraint.num_rows
    assert constraint.geometry.r is not r


def test_arm_rows_have_signed_distance_jacobians_and_protected_columns():
    r=Robot();constraint=ArmGeometry(object(),r,50.)
    row=np.r_[0.,0.,1.,np.zeros(29)]
    for g in r.sides.values():row[g['arm'][3]]=-.4
    G,h=constraint.measure(row,np.eye(3))
    for j in r.sides['Left']['arm'][:3]:
        step=np.zeros(32);step[j]=1e-6
        numeric=(constraint.measure(row+step,np.eye(3))[1]-constraint.measure(row-step,np.eye(3))[1])/2e-6
        np.testing.assert_allclose(numeric,-G[:,j],atol=3e-5)
    mask=np.ones(32,bool);mask[sum([g['arm'] for g in r.sides.values()],[])]=False
    proposal=row.copy();proposal[3:15]=.2
    values=jax.jit(constraint.rows)(jnp.asarray(proposal),dict(root_r=jnp.eye(3),priority_mask=jnp.asarray(mask),priority_pose=jnp.asarray(row)))
    np.testing.assert_array_equal(np.asarray(values[0])[:,mask],0.)
    np.testing.assert_allclose(values[1],50*h)


def test_arm_geometry_participates_in_runtime_kernel_identity():
    from dex_frax.ik import runtime_keys
    r=Robot();n=2;rot=np.tile(np.eye(3),(n,1,1));vector=np.zeros((n,3))
    sides={}
    for s in r.sides:
        sides[s]={k:vector.copy() for k in ['thigh','shin','foot_target_p','pivots','anchors','upper','lower',
                  'hand_forward','hand_width','forearm_width','elbow_axis']}
        sides[s].update(foot_target_r=rot.copy(),contact=np.zeros(n),signed_elbow_bend=np.zeros(n),ground_contact_confidence=np.zeros(n))
    t=dict(root_p=vector,root_r=rot,torso_r=rot,hips=vector,fps=50.,sides=sides,
           body_contact_height_tolerance_m=.035,body_contacts=[])
    a=runtime_keys(r,t,{},8,100,False,{})
    b=runtime_keys(r,dict(t,arm_self_collision=True),{},8,100,False,{})
    assert a[0]==b[0] and a[1]!=b[1]


def test_swept_rows_use_endpoint_derivative_and_match_root_slerp_distances():
    from scipy.spatial.transform import Rotation
    r=Robot();con=ArmGeometry(object(),r,50.);rng=np.random.default_rng(21)
    previous=np.r_[0.,0.,1.,np.zeros(29)];current=previous.copy()
    ids=sum([g['arm'] for g in r.sides.values()],[])
    previous[ids]=rng.uniform(-.3,.3,len(ids));current[ids]=previous[ids]+rng.uniform(-.05,.05,len(ids))
    rotation=Rotation.from_rotvec([.3,-.2,.4]).as_matrix()
    G,h=con.measure_swept(current,rotation,previous,False)
    for j in ids[:3]:
        delta=np.zeros(32);delta[j]=1e-6
        numeric=(con.measure_swept(current+delta,rotation,previous,False)[1]-con.measure_swept(current-delta,rotation,previous,False)[1])/2e-6
        np.testing.assert_allclose(numeric,-G[:,j],atol=4e-5)
    start=Rotation.from_rotvec([-.2,.1,-.3]).as_matrix()
    difference=Rotation.from_matrix(start.T@rotation).as_rotvec()
    for alpha in np.arange(1,9)/8:
        row=previous*(1-alpha)+current*alpha
        con.geometry.update(row,rotation);a=con.geometry.distances()
        con.geometry.update(row,start@Rotation.from_rotvec(difference*alpha).as_matrix());b=con.geometry.distances()
        np.testing.assert_allclose(a,b,atol=1e-12)


def test_reduced_columns_retain_every_collision_row_and_free_derivative():
    from dex_frax.ik import solve_columns,HingeSpace,FrameBounds
    r=Robot();ids=np.asarray(sum([g['arm'] for g in r.sides.values()],[]))
    mask=np.ones(32,bool);mask[ids]=False
    t=dict(arm_self_collision=True,priority_mask=mask)
    np.testing.assert_array_equal(solve_columns(r,t),ids)
    for other in (dict(t,foot_self_collision=True),dict(t,arm_self_collision=False),
                  dict(t,priority_mask=np.zeros(32,bool))):
        np.testing.assert_array_equal(solve_columns(r,other),np.arange(32))
    full=ArmGeometry(object(),r,50.);small=ArmGeometry(HingeSpace(len(ids)),r,50.,columns=ids)
    q=np.r_[0.,0.,1.,np.zeros(29)];q[ids[3]]=-.4
    current=q.copy();current[ids[0]]+=.02
    a,h=full.measure_swept(current,np.eye(3),q,False)
    b,k=small.measure_swept(current,np.eye(3),q,False)
    assert small.num_rows==full.num_rows
    np.testing.assert_array_equal(a[:,ids],b);np.testing.assert_array_equal(h,k)
    rows=jax.jit(small.rows)(jnp.asarray(current[ids]),dict(root_r=jnp.eye(3),
        priority_mask=jnp.asarray(mask),priority_pose=jnp.asarray(current),previous=jnp.asarray(q),first=0.))
    np.testing.assert_allclose(rows[0],b);np.testing.assert_allclose(rows[1],50.*k)
    step=r.joint_step(50.,{'numerical_joint_speed_deg_s':720.})
    large=FrameBounds(type('Space',(),{'nv':32})(),r.lower[3:],r.upper[3:],step,50.)
    reduced=FrameBounds(HingeSpace(len(ids)),r.lower[ids],r.upper[ids],step[ids-3],50.,previous_columns=ids)
    for first in (False,True):
        G,v=large.rows(current,dict(previous=q,first=first));H,w=reduced.rows(current[ids],dict(previous=q,first=first))
        owned=np.r_[ids-3,29+ids-3]
        np.testing.assert_array_equal(G[owned][:,ids],H);np.testing.assert_array_equal(np.asarray(v)[owned],w)


def test_fast_measurement_keeps_all_eight_auditor_clearances():
    r=Robot();con=ArmGeometry(object(),r,50.)
    previous=np.r_[.2,-.1,1.,np.zeros(29)]
    ids=sum([g['arm'] for g in r.sides.values()],[])
    previous[ids]=np.random.default_rng(37).uniform(-.3,.3,len(ids))
    q=previous.copy();q[ids]+=.03
    _,gap=con.measure_swept(q,np.eye(3),previous,False)
    expected=[]
    for alpha in np.arange(1,9)/8:
        con.geometry.update(previous*(1-alpha)+q*alpha,np.eye(3))
        expected.append(con.geometry.distances())
    minimum=np.min(expected,axis=0);near=minimum<con.geometry.cfg['detection_m']-1e-12
    np.testing.assert_allclose(gap[near],minimum[near]-con.margin,atol=1e-10)
    np.testing.assert_array_equal(gap[~near],1.)


def test_recovery_preference_is_absent_on_approach_and_decays_after_release():
    from dex_frax.frax_arm_constraints import release_activity
    memory=jnp.zeros(2)
    for proximity in (jnp.array([.3,.5]),jnp.array([.8,.9]),jnp.ones(2)):
        memory,weight=release_activity(proximity,memory,.8)
        np.testing.assert_array_equal(weight,0.)
    memory,weight=release_activity(jnp.zeros(2),memory,.8)
    np.testing.assert_allclose(weight,.8)
    memory,weight=release_activity(jnp.zeros(2),memory,.8)
    np.testing.assert_allclose(weight,.64)
    _,weight=release_activity(jnp.ones(2),memory,.8)
    np.testing.assert_array_equal(weight,0.)


def test_recovery_dead_zone_uses_neighbouring_source_region_motion():
    from dex_frax.frax_arm_constraints import source_arm_steps
    angle=np.array([0.,.02,.06,.08,.08]);vectors=np.c_[np.cos(angle),np.sin(angle),np.zeros(5)]
    still=np.tile([1.,0.,0.],(5,1));keys=('upper','lower','hand_forward','hand_width','elbow_axis')
    t=dict(root_p=np.zeros((5,3)),sides=dict(Left={k:vectors for k in keys},Right={k:still for k in keys}))
    result=source_arm_steps(t)
    np.testing.assert_allclose(result[:,0],[.04,.08,.08,.08,.04],atol=1e-12)
    np.testing.assert_allclose(result[:,1],np.deg2rad(1.))
    swapped=dict(t,sides=dict(reversed(list(t['sides'].items()))))
    np.testing.assert_array_equal(source_arm_steps(swapped),result)
