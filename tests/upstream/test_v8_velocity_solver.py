"""Physical speed limits affect generation as well as output acceptance."""
import numpy as np
import pytest
from dex_frax.v8.robot import Robot
from dex_frax.v8.trajectory import project_velocity
from dex_frax.quality_v8 import adjacent_speed_report


def require_reference_surface(name):
    from dex_frax.v8.source import REFERENCE
    if not (REFERENCE / name).is_file():
        pytest.skip(f'Licensed reference table {name} required; see docs/INPUTS.md')


@pytest.fixture
def lafan_surface():
    require_reference_surface('lafan1_surfaces.json')


@pytest.fixture
def soma_surface():
    require_reference_surface('soma_surfaces.json')


def test_step_uses_named_motor_limits_and_retains_numerical_ceiling():
    r=Robot.__new__(Robot)
    r.speed_limits=np.array([7.,11.1,16.])
    step=r.joint_step(50.,{'numerical_joint_speed_deg_s':720.})
    np.testing.assert_allclose(step,[7/50,11.1/50,np.deg2rad(720)/50])


def test_projection_enforces_each_axis_without_changing_sample_count():
    q=np.array([[0.,0.],[.6,-.6],[-.6,.6],[.5,-.5],[0.,0.]])
    limits=np.array([2.,5.]);step=limits/50
    out=project_velocity(q,step,np.array([-.4,-.5]),np.array([.4,.5]))
    assert out.shape==q.shape
    assert np.all(out>=[-.4,-.5]) and np.all(out<=[.4,.5])
    assert not np.array_equal(out,q)
    assert not adjacent_speed_report(out.astype(np.float32),50.,['slow','fast'],limits)['violating_intervals']


def test_already_feasible_trajectory_is_preserved():
    q=np.array([[0.,0.],[.01,.02],[.02,.01]])
    np.testing.assert_array_equal(project_velocity(q,np.array([.03,.03]),[-1,-1],[1,1]),q)


def test_long_transition_projection_is_feasible_and_preserves_end_regions():
    # A long saturated run defeats the old fixed 1200 pair-sweep budget.
    q=np.r_[np.zeros(1000),np.ones(1000)*2.5][:,None]
    out=project_velocity(q,np.array([.002]),np.array([-3.]),np.array([3.]))
    assert np.max(abs(np.diff(out[:,0]))) <= .002+1e-9
    assert abs(out[0,0]) < .01 and abs(out[-1,0]-2.5) < .01
    expected=np.clip(1.25+(np.arange(2000)-999.5)*(.002*(1.-1e-5)),0.,2.5)
    np.testing.assert_allclose(out[:,0],expected,atol=1e-5)


def test_constrained_pose_update_coordinates_axes_to_keep_end_direction():
    from scipy.sparse import csr_matrix
    from dex_frax.v8.trajectory import constrained_refine
    n=25
    target=np.r_[np.zeros(5),np.linspace(0,1.,15),np.ones(5)]
    q=np.column_stack([target,np.zeros(n)])
    def fun(z):
        v=z.reshape(n,2)
        return np.r_[v.sum(axis=1)-target,.01*v[:,1]]
    J=np.zeros((2*n,2*n))
    for i in range(n):J[i,2*i:2*i+2]=1.;J[n+i,2*i+1]=.01
    out,report=constrained_refine(q,fun,lambda z:csr_matrix(J),[-2,-2],[2,2],[.01,.1])
    assert np.max(abs(out.sum(axis=1)-target)) < .001
    assert np.all(abs(np.diff(out,axis=0)) <= np.array([.01,.1])+1e-9)
    assert report['cost'] < report['initial_cost']*.01
    assert np.max(abs(out[:,1])) > .2


def test_local_review_finds_large_short_loss_hidden_by_whole_clip_p95():
    from dex_frax.v8.review import compare_local_errors
    old=np.ones(10000);new=old.copy();new[2000:2030]=30.
    assert np.percentile(new,95)==1.
    result=compare_local_errors(old,new,50.)
    assert result['material_window_count']>0
    assert result['review_windows'][0]['new_p95_deg']==30.
    small=old.copy();small[2000:2030]=5.
    assert compare_local_errors(old,small,50.)['material_window_count']==0


def test_fixed_source_foot_profile_needs_no_standing_or_ground_observation(lafan_surface):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from dex_frax.v8.source_profile import lafan_foot
    n=100
    motion=SimpleNamespace(names=['LeftFoot','LeftToe'],parents=np.array([-1,0]),offsets=np.array([[0,0,0],[.173,0,0]]))
    foot=Rotation.from_euler('x',np.linspace(0,360,n)[:,None],degrees=True).as_matrix()
    toe=foot@Rotation.from_euler('z',21.454,degrees=True).as_matrix()
    p={'LeftFoot':np.tile([0.,0.,1.],(n,1)),'LeftToe':np.tile([0.,0.,1.],(n,1))+foot@np.array([.173,0,0])}
    r={'LeftFoot':foot,'LeftToe':toe}
    frame,heel,tip,record,scene=lafan_foot(motion,p,r,'Left')
    assert record['standing_frames_required'] is False and scene['height_m']==0.
    assert np.isfinite(frame).all() and np.isfinite(heel).all()
    # Sole orientation follows the rotating foot, including a full inversion.
    local=np.einsum('nji,njk->nik',foot,frame)
    np.testing.assert_allclose(local,np.broadcast_to(local[0],local.shape),atol=1e-10)


def test_contact_modes_distinguish_fixed_sliding_rolling_and_airborne():
    from dex_frax.v8.multicontact import classify_contact
    from scipy.spatial.transform import Rotation
    n=200;fps=50.;cfg=dict(height_tolerance_m=.035,speed_enter=.1,speed_leave=.2,stable_duration_s=.1)
    R=np.broadcast_to(np.eye(3),(n,3,3));fixed=np.zeros((n,3))
    assert np.all(classify_contact(fixed,R,np.zeros(n),fps,cfg)==1)
    sliding=fixed.copy();sliding[:,0]=np.arange(n)/fps*.3
    assert np.all(classify_contact(sliding,R,np.zeros(n),fps,cfg)==2)
    rotating=Rotation.from_euler('y',(np.arange(n)/fps)[:,None]).as_matrix()
    assert np.all(classify_contact(sliding,rotating,np.zeros(n),fps,cfg)[1:]==3)
    assert not classify_contact(fixed,R,np.ones(n),fps,cfg).any()


def test_head_contact_region_matches_complete_fixed_head_after_body_rotation():
    from dex_frax.v8.multicontact import region_hull,contact_local
    from dex_frax.paths import ASSETS
    from scipy.spatial.transform import Rotation
    if not (ASSETS/'mjcf/dex_29.xml').exists():pytest.skip('Local robot assets required')
    robot=Robot();body=robot.bid('head_yaw_link');hull=region_hull(robot,'head',body)
    contact=dict(body=body,robot_hull=hull)
    # The semantic head must include the shell/camera, rather than a neck motor.
    assert len(hull)>len(robot.surface_models[robot.bid('head_pitch_link')]['hull'])
    for angle in [0.,.5,1.5,3.]:
        x=np.r_[0.,0.,.3,np.zeros(29)];x[3+robot.names.index('waist_pitch_joint')]=.2
        robot.set(x,Rotation.from_euler('x',angle).as_matrix())
        expected=min(robot.point(b)[2]+np.min(g['hull']@robot.rot(b)[2]) for b,g in robot.surface_models.items() if 'head' in g['name'] and robot.m.body_weldid[b]==robot.m.body_weldid[body])
        observed=robot.point(body,contact_local(robot,contact))[2]
        assert observed==pytest.approx(expected,abs=1e-9)


def test_limb_contact_surfaces_cover_the_limb_not_only_its_joint_motor():
    from dex_frax.paths import ASSETS
    if not (ASSETS/'mjcf/dex_29.xml').exists():pytest.skip('Local robot assets required')
    r=Robot()
    for side,suffix in [('Left','l'),('Right','r')]:
        g=r.sides[side]
        for name,a,b in [('hip_yaw_'+suffix+'_link','hip','knee'),('knee_pitch_'+suffix+'_link','knee','foot'),('shoulder_yaw_'+suffix+'_link','shoulder','elbow'),('elbow_yaw_'+suffix+'_link','elbow','wrist')]:
            body=r.bid(name);delta=r.point(g[b])-r.point(g[a]);length=np.linalg.norm(delta);axis=delta/length
            world=r.surface_models[body]['hull']@r.rot(body).T+r.point(body)
            coverage=np.ptp(world@axis)
            assert coverage>.6*length,(side,name,coverage,length)


def test_standing_foot_anchors_survive_and_whole_clip_flight_is_not_grounded(soma_surface):
    import json
    from pathlib import Path
    from dex_frax.v8.source import adapt,REFERENCE
    from dex_frax.v8.bvh import load
    from dex_frax.v8.contacts import prepare
    from dex_frax.v8 import contacts
    from dex_frax.paths import ASSETS
    if not (ASSETS/'mjcf/dex_29.xml').exists():pytest.skip('Local robot assets required')
    cfg=json.loads(Path(contacts.__file__).with_name('config.json').read_text())['contact']
    motion=load(REFERENCE/'soma_zero.bvh')
    motion.local_p=np.repeat(motion.local_p[:1],100,axis=0)
    motion.local_r=np.repeat(motion.local_r[:1],100,axis=0)
    robot=Robot();standing=prepare(adapt(motion,'soma'),robot,cfg)
    for side,v in standing['sides'].items():
        assert np.all(v['phase']==1)
        np.testing.assert_allclose(v['anchors'],np.broadcast_to(v['anchors'][0],v['anchors'].shape))
        g=robot.sides[side]
        lowest=v['foot_target_p'][:,2]+np.min(v['foot_target_r'][:,2,:]@g['hull'].T,axis=1)
        np.testing.assert_allclose(lowest,0.,atol=1e-12)
    # The same rig entirely above the scene is airborne, not a new floor.
    motion.local_p[:,0,1]+=1.
    flight=prepare(adapt(motion,'soma'),robot,cfg)
    assert flight['source_airborne'].all()
    assert not any(v['contact'].any() for v in flight['sides'].values())
    assert not any(v['mode'].any() for v in flight['body_contacts'])
    np.testing.assert_allclose(flight['root_p'][:,2]-standing['root_p'][:,2],standing['scale'])


def test_long_lafan_shoe_uses_same_flat_angle_without_flattening_a_raised_heel(monkeypatch):
    import json
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from dex_frax.v8.source import adapt,REFERENCE
    from dex_frax.v8.bvh import load
    from dex_frax.v8 import contacts,multicontact
    from dex_frax.paths import ASSETS
    if not (ASSETS/'mjcf/dex_29.xml').exists():pytest.skip('Local robot assets required')
    # Isolate the foot-state gate from unrelated source body-surface names.
    monkeypatch.setattr(multicontact,'prepare',lambda t,*a:t)
    cfg=json.loads(Path(contacts.__file__).with_name('config.json').read_text())['contact']
    motion=load(REFERENCE/'soma_zero.bvh');motion.local_p=np.repeat(motion.local_p[:1],100,axis=0);motion.local_r=np.repeat(motion.local_r[:1],100,axis=0)
    def scenario(heel_z):
        t=adapt(motion,'soma');t['source_type']='lafan'
        for v in t['sides'].values():
            v['heel'][:]=[0.,0.,heel_z];v['toe'][:]=[.30,0.,-.03]
            v['foot_r'][:]=Rotation.from_euler('y',np.deg2rad(10.5)).as_matrix()
        return contacts.prepare(t,Robot(),cfg)
    quiet=scenario(.025)
    assert all(np.all(v['phase']==1) for v in quiet['sides'].values())
    raised=scenario(.06)
    assert all(np.all(v['phase']==2) for v in raised['sides'].values())


def test_contact_feasibility_repair_preserves_plane_anchor_and_joint_bounds():
    from scipy.sparse import csc_matrix
    from dex_frax.v8.contact_solver import feasible_projection
    # A micron-scale primal stall is not a reason to loosen the floor, the
    # fixed horizontal anchor or the motor's admissible update interval.
    A=csc_matrix([[1.,0.],[1.,1e-5],[0.,1.],[1.,0.]])
    lo=np.array([4e-6,4e-6,0.,-.1]);hi=np.array([np.inf,np.inf,0.,.1])
    point,info=feasible_projection(np.array([2e-6,0.]),A,lo,hi)
    np.testing.assert_allclose(point,[4e-6,0.],atol=1e-9)
    assert np.all(A@point>=lo-1e-9) and np.all(A@point<=hi+1e-9)
    assert info['projection_change']==pytest.approx(2e-6,abs=1e-9)


def test_contact_feasibility_repair_rejects_incompatible_hard_rows():
    from scipy.sparse import csc_matrix
    from dex_frax.v8.contact_solver import feasible_projection
    with pytest.raises(ValueError,match='infeasible'):
        feasible_projection(np.zeros(1),csc_matrix([[1.],[1.]]),np.array([1.,-np.inf]),np.array([np.inf,0.]))


def test_whole_body_clearance_does_not_conflict_with_planted_sole_zero(monkeypatch):
    from dex_frax.v8 import contact_solver
    from dex_frax.paths import ASSETS
    if not (ASSETS/'mjcf/dex_29.xml').exists():pytest.skip('Local robot assets required')
    robot=Robot();base=np.r_[0.,0.,1.,np.zeros(29)]
    robot.set(base,np.eye(3));base[2]-=min(robot.point(g['foot'],robot.lowest(s))[2] for s,g in robot.sides.items())
    robot.set(base,np.eye(3));n=3
    t=dict(fps=50.,root_r=np.tile(np.eye(3),(n,1,1)),body_contacts=[dict(mode=np.ones(n))],sides={})
    for s,g in robot.sides.items():
        anchor=robot.point(g['foot'],g['sole']);anchor[2]=0.
        t['sides'][s]=dict(contact=np.ones(n),pivots=np.tile(g['sole'],(n,1)),anchors=np.tile(anchor,(n,1)))
    class HoldPose:
        def __init__(self,*args):pass
        def evaluate(self,x):return x-base,np.eye(32)
    monkeypatch.setattr(contact_solver,'Objective',HoldPose)
    inputs=np.tile(base,(n,1));inputs[1,robot.sides['Left']['arm'][0]]=.4
    out,reports=contact_solver.refine(robot,t,inputs,dict(numerical_joint_speed_deg_s=720.))
    assert not reports
    assert robot.body_contact_qp_summary['steps']>0
    assert robot.body_contact_qp_summary['normalized_retries']==0
    assert robot.body_contact_qp_summary['feasibility_projections']==0
    assert robot.body_contact_qp_summary['maximum_constraint_excess']<1e-8
    for i,row in enumerate(out):
        robot.set(row,t['root_r'][i])
        for side,g in robot.sides.items():
            # Check the frame that actually required a constrained update;
            # the neutral mesh shoes initially differ by about 24 microns.
            if i==1:assert abs(robot.point(g['foot'],robot.lowest(side))[2])<1e-6
            np.testing.assert_allclose(robot.point(g['foot'],g['sole'])[:2],t['sides'][side]['anchors'][i,:2],atol=1e-6)


def test_contact_hand_priority_keeps_aerial_weights_and_uses_existing_arm_scale():
    from dex_frax.v8.task_weights import hand_weights
    cfg=dict(arm_direction=4.,hand_direction=3.,hand_width=2.)
    assert hand_weights(cfg,{},0)==(3.,2.)
    side=dict(ground_contact_confidence=np.array([0.,.5,1.]))
    np.testing.assert_allclose([hand_weights(cfg,side,i) for i in range(3)],[[3,2],[3.5,3],[4,4]])


def test_arm_floor_and_contact_rows_have_correct_real_model_derivatives(soma_surface):
    import json
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from dex_frax.v8.source import adapt,REFERENCE
    from dex_frax.v8.bvh import load
    from dex_frax.v8.contacts import prepare
    from dex_frax.v8.arms import ArmObjective
    from dex_frax.v8.anatomy import configure
    from dex_frax.v8 import arms
    from dex_frax.paths import ASSETS
    if not (ASSETS/'mjcf/dex_29.xml').exists():pytest.skip('Local robot assets required')
    cfg=json.loads(Path(arms.__file__).with_name('config.json').read_text())
    robot=Robot();configure(robot,cfg['anatomy']);motion=load(REFERENCE/'soma_zero.bvh')
    motion.local_p=np.repeat(motion.local_p[:1],30,axis=0);motion.local_r=np.repeat(motion.local_r[:1],30,axis=0)
    t=prepare(adapt(motion,'soma'),robot,cfg['contact']);t['root_r'][:]=Rotation.from_euler('xyz',[.2,.1,.3]).as_matrix()
    for c in t['body_contacts']:
        if c['key']=='Left_hand':c['mode'][:]=1;c['confidence'][:]=1.
    g=robot.sides['Left'];base=np.r_[0.,0.,-.6,np.zeros(29)];q=np.array([-.4,.3,.2,-.9,.4,.1,.2]);base[g['arm']]=q
    objective=ArmObjective(robot,t,0,'Left',base,cfg['solver'])
    value,J=objective.evaluate(q)
    # Three major arm surfaces plus three inferred support rows are included.
    assert np.any(value[-6:-3]<0)
    assert value[-1]<0 and np.linalg.norm(J[-1])>0
    eps=1e-6;numeric=np.column_stack([(objective.fun(q+np.eye(7)[j]*eps)-objective.fun(q-np.eye(7)[j]*eps))/(2*eps) for j in range(7)])
    np.testing.assert_allclose(J,numeric,atol=3e-5,rtol=1e-4)
