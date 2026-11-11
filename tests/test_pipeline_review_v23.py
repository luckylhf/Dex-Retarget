import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation
from dex_frax.v8.robot import Robot
from dex_frax.ik import solve_columns,HingeSpace,FrameBounds
from dex_frax.frax_foot_constraints import FootGeometry,FootSupport
from dex_frax.kinematics import DexKinematics
from dex_frax.refine_queue import WorkQueue,jobs_from_masks
from dex_frax.refine_routing import ready_jobs


def test_root_height_leg_elimination_preserves_every_hinge_bound_and_collision_row():
    r=Robot();legs=np.asarray(sum([g['leg'] for g in r.sides.values()],[]));ids=np.r_[2,legs]
    mask=np.ones(32,bool);mask[ids]=False
    np.testing.assert_array_equal(solve_columns(r,dict(foot_self_collision=True,priority_mask=mask)),ids)
    step=r.joint_step(50.,{'numerical_joint_speed_deg_s':720.})
    large=FrameBounds(HingeSpace(32),r.lower[3:],r.upper[3:],step,50.)
    small=FrameBounds(HingeSpace(13),r.lower[legs],r.upper[legs],step[legs-3],50.,previous_columns=legs,local_columns=np.arange(1,13))
    previous=np.r_[0.,0.,.9,np.zeros(29)];row=previous.copy();row[ids]+=.01
    G,h=large.rows(row,dict(previous=previous,first=False));A,b=small.rows(row[ids],dict(previous=previous,first=False))
    owned=np.r_[legs-3,29+legs-3]
    np.testing.assert_allclose(G[owned][:,ids],A);np.testing.assert_allclose(np.asarray(h)[owned],b)
    assert A.shape==(24,13) and not np.any(A[:,0])
    full=FootGeometry(object(),r,50.);reduced=FootGeometry(HingeSpace(13),r,50.,columns=ids)
    G,h=full.measure(row,np.eye(3));A,b=reduced.measure(row,np.eye(3))
    np.testing.assert_allclose(G[:,ids],A);np.testing.assert_allclose(h,b)


def test_reduced_foot_support_uses_root_z_and_identical_sole_and_anchor_rows():
    r=Robot();model=DexKinematics(r);legs=sum([g['leg'] for g in r.sides.values()],[]);ids=np.r_[2,legs];mask=np.ones(32,bool);mask[ids]=False
    row=np.r_[.1,-.2,.9,np.zeros(29)];r.set(row,np.eye(3));sides={}
    for side,g in r.sides.items():sides[side]=dict(contact=1.,pivots=g['sole'],anchors=r.point(g['foot'],g['sole']))
    ref=dict(root_r=jnp.eye(3),priority_pose=jnp.asarray(row),priority_mask=jnp.asarray(mask),sides=jax.tree.map(jnp.asarray,sides))
    full=FootSupport(model,r,50.);reduced=FootSupport(model,r,50.,columns=ids)
    G,h=jax.jit(full.rows)(jnp.asarray(row),ref);A,b=jax.jit(reduced.rows)(jnp.asarray(row[ids]),ref)
    np.testing.assert_allclose(G[:,ids],A,atol=1e-10);np.testing.assert_allclose(h,b,atol=1e-10)
    assert np.any(abs(A[:,0])>0)


def test_waist_solution_does_not_depend_on_root_translation_or_leg_seed():
    from dex_frax.pelvis_priority import waist_trajectory
    r=Robot();n=4;row=np.r_[0.,0.,1.,np.zeros(29)];seed=np.tile(row,(n,1))
    t=dict(fps=50.,root_r=np.tile(np.eye(3),(n,1,1)),torso_r=Rotation.from_euler('xyz',np.array([[.01,.02,0.],[.02,.01,.01],[.01,.01,.02],[0.,.02,.01]])).as_matrix())
    first,ids=waist_trajectory(r,t,seed,{'numerical_joint_speed_deg_s':720.})
    other=seed.copy();other[:,:3]+=[.4,-.2,.3];other[:,sum([g['leg'] for g in r.sides.values()],[])]=.2
    second,_=waist_trajectory(r,t,other,{'numerical_joint_speed_deg_s':720.})
    np.testing.assert_allclose(first[:,ids],second[:,ids],atol=1e-8)


def test_normal_and_scoped_safety_jobs_have_one_fair_unblocked_selector():
    q=WorkQueue();n=200;mask=np.arange(n)==80
    pending=jobs_from_masks([('support',None,mask),('jitter','hip_yaw_l_joint',mask),('leg_collision',None,mask),('arms','Left',mask)],n)
    zero=np.zeros(n);clear=np.zeros(n,bool)
    normal,_=ready_jobs(q,pending,zero,clear,clear,[])
    scoped,_=ready_jobs(q,[j for j in pending if j['module']!='arms'],zero,clear,clear,[])
    assert normal==scoped and {j['module'] for j in normal}=={'leg_collision'}
    next_jobs,_=ready_jobs(q,pending,zero,clear,clear,[{'module':'leg_collision'}])
    assert {j['module'] for j in next_jobs}=={'support'}
    # Exhausting support cannot block a coupled leg fit that also repairs it.
    support=next(j for j in pending if j['module']=='support');state=q.state(support)
    state['done']=q.choices(support)
    ready,_=ready_jobs(q,pending,zero,clear,clear,[])
    assert {j['module'] for j in ready}=={'leg_collision'}


def test_real_reduced_initial_solver_registers_support_and_executes_finite_steps():
    import json
    from pathlib import Path
    from dex_frax.ik import FraxIK
    from dex_frax.v8.anatomy import configure
    from dex_frax.v8 import convert
    r=Robot();cfg=json.loads((Path(convert.__file__).parent/'config.json').read_text());configure(r,cfg['anatomy'])
    x=np.r_[0.,0.,.9,np.zeros(29)]
    for g in r.sides.values():x[g['leg'][3]]=.15;x[g['arm'][3]]=-.4
    r.set(x,np.eye(3));height=min(float((g['hull']@r.rot(g['foot'])[2]+r.point(g['foot'])[2]).min()) for g in r.sides.values());x[2]-=height;r.set(x,np.eye(3))
    n=3;repeat=lambda v:np.repeat(np.asarray(v)[None],n,axis=0);sides={}
    for side,g in r.sides.items():
        unit=lambda a,b:(r.point(g[b])-r.point(g[a]))/np.linalg.norm(r.point(g[b])-r.point(g[a]))
        sides[side]={k:repeat(v) for k,v in dict(thigh=unit('hip','knee'),shin=unit('knee','foot'),foot_target_p=r.point(g['foot']),foot_target_r=r.rot(g['foot']),pivots=g['sole'],anchors=r.point(g['foot'],g['sole']),upper=unit('shoulder','elbow'),lower=unit('elbow','wrist'),hand_forward=-r.rot(g['tcp'])[:,2],hand_width=r.rot(g['tcp'])[:,0],forearm_width=r.rot(g['forearm'])[:,0],elbow_axis=-r.rot(g['shoulder'])[:,1],signed_elbow_bend=g['elbow_geometric_zero']-x[g['arm'][3]],ground_contact_confidence=0.,contact=1.).items()}
    mask=np.ones(32,bool);mask[np.r_[2,sum([g['leg'] for g in r.sides.values()],[])]]=False
    t=dict(fps=50.,root_p=repeat(x[:3]),root_r=repeat(np.eye(3)),torso_r=repeat(r.rot(r.torso)),hips=repeat(x[:3]),sides=sides,body_contacts=[],body_contact_height_tolerance_m=.035,priority_pose=repeat(x),priority_mask=mask,foot_self_collision=True)
    solver=FraxIK(1,2,improved=False,use_cache=False,block_frames=4);out,records=solver(r,t,cfg['solver'])
    assert out.shape==(n,32) and np.isfinite(out).all() and len(records)==n
    assert solver.report['solve_coordinate_count']==13
    np.testing.assert_array_equal(out[:,mask],repeat(x)[:,mask])
