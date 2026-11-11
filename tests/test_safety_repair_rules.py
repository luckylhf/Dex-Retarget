import numpy as np
from scipy.sparse import eye
from dex_frax.temporal_quality import assess,POLICY
from dex_frax.temporal_constraints import corridor,diagnostics,regressions
from dex_frax.v8.trajectory import _qp
from dex_frax.v8.robot import Robot
from dex_frax.v8.self_collision import CollisionModel
from dex_frax.refine_routing import collision_groups
from dex_frax.frax_foot_constraints import FootGeometry


def source(n):
    still=np.tile([1.,0.,0.],(n,1));rr=np.tile(np.eye(3),(n,1,1))
    return dict(fps=50.,root_r=rr,torso_r=rr,sides={s:{k:still.copy() for k in ('upper','lower','hand_forward','hand_width','elbow_axis','thigh','shin')} for s in ('Left','Right')})


def test_source_supported_corridor_removes_a_velocity_pulse_without_new_gate():
    r=Robot();n=12;t=source(n);x=np.zeros((n,32));j=r.sides['Right']['arm'][5];x[5:,j]=np.deg2rad(3.)
    original=dict(POLICY);before=assess(x[:,3:],50.,r.names,diagnostics(t));assert before['joints'][r.names[j-3]]['violating_frames']
    A,b,report=corridor(r,t,x,0,n,[j]);q,_=_qp(eye(n,format='csc'),-x[:,j],A,-b,b,x[:,j],scaling=10)
    after=x.copy();after[:,j]=q
    assert not assess(after[:,3:],50.,r.names,diagnostics(t))['violating_frames']
    assert POLICY==original and report['targeted_rows']>0 and not report['hardware_acceleration_limit']


def test_jitter_cannot_move_to_another_joint_under_the_same_union_frame():
    old={'joints':{'ankle':{'violating_frames':[10,11]}},'violating_frames':[10,11]}
    new={'joints':{'ankle':{'violating_frames':[]},'wrist':{'violating_frames':[10,11]}},'violating_frames':[10,11]}
    assert regressions(old,new)=={'wrist':[10,11]}


def test_initial_leg_constraints_cover_the_entire_declared_leg_class():
    r=Robot();full=CollisionModel(r,{'coverage':'full'});owned=collision_groups(full)['leg_collision'];constraint=FootGeometry(object(),r,50.)
    assert set(map(tuple,constraint.geometry.pairs))==set(map(tuple,full.pairs[owned]))
    assert ['knee_pitch_l_link_collision','knee_pitch_r_link_collision'] in constraint.geometry.names or ['knee_pitch_r_link_collision','knee_pitch_l_link_collision'] in constraint.geometry.names
    assert constraint.interpolation_steps==8


def test_local_swept_floor_rows_couple_both_samples_with_correct_derivative():
    from dex_frax.local_geometry import Geometry
    r=Robot();n=5;rows=np.tile(np.r_[0.,0.,1.,np.zeros(29)],(n,1));r.set(rows[0],np.eye(3))
    bottom=min(float((g['hull']@r.rot(g['foot'])[2]+r.point(g['foot'])[2]).min()) for g in r.sides.values());rows[:,2]-=bottom+.004
    rows[:,2]+=np.array([.001,.0002,.0007,0.,.0009]);t=source(n);t['body_contacts']=[]
    for side,g in r.sides.items():t['sides'][side].update(contact=np.zeros(n),pivots=np.tile(g['sole'],(n,1)),anchors=np.zeros((n,3)))
    geometry=Geometry(r,t,rows,0,n,np.array([2]),lambda b:True,restore=True)
    A,lo,hi=geometry.evaluate(rows[:,2],linear=True);dense=np.asarray(A.toarray());assert np.any(np.sum(abs(dense)>1e-12,axis=1)==2)
    dz=np.arange(n)*1e-8;_,shifted,_=geometry.evaluate(rows[:,2]+dz,linear=True)
    np.testing.assert_allclose(shifted-lo,-A@dz,atol=2e-9)


def test_cached_clearance_shortcut_requires_both_exact_coverages_and_no_negative_minimum():
    from dex_frax.audit_cache import clear_dense_window
    n=7;cache={'full_collision':{'samples':[(.01,0,[]) for _ in range((n-1)*8+1)]},'floor':np.zeros(((n-1)*8+1,2))}
    assert clear_dense_window(cache,2,5,n)
    cache['full_collision']['samples'][3*8+5]=(-1e-8,0,[])
    assert not clear_dense_window(cache,2,5,n)
    assert clear_dense_window(cache,0,2,n)
    assert not clear_dense_window({'full_collision':cache['full_collision']},0,2,n)


def test_support_ball_allows_safe_diagonal_motion_and_rejects_outside_radius():
    from dex_frax.local_geometry import Geometry
    r=Robot();n=5;rows=np.tile(np.r_[0.,0.,1.,np.zeros(29)],(n,1));t=source(n);t['body_contacts']=[]
    r.set(rows[0],np.eye(3));foot=r.sides['Left'];height=(foot['hull']@r.rot(foot['foot'])[2]+r.point(foot['foot'])[2]).min();rows[:,2]-=height;r.set(rows[0],np.eye(3))
    for side,g in r.sides.items():
        anchor=r.point(g['foot'],g['sole']);anchor[2]=0
        t['sides'][side].update(contact=np.ones(n) if side=='Left' else np.zeros(n),pivots=np.tile(g['sole'],(n,1)),anchors=np.tile(anchor,(n,1)))
    geometry=Geometry(r,t,rows,0,n,np.array([0,1]),lambda b:True)
    z=rows[:,:2].copy();A,lo,hi=geometry.evaluate(z.ravel(),linear=True)
    delta=np.tile([.0006,.0005],(n,1));at=A@delta.ravel()
    assert np.all(at>=lo-1e-10) and np.all(at<=hi+1e-10)
    assert geometry.evaluate((z+delta).ravel())<1e-10
    assert geometry.evaluate((z+.0008).ravel())>0.


def test_passing_contact_near_gate_keeps_interior_room_for_smoothing():
    from dex_frax.local_geometry import Geometry
    r=Robot();n=5;rows=np.tile(np.r_[0.,0.,1.,np.zeros(29)],(n,1));t=source(n);t['body_contacts']=[]
    r.set(rows[0],np.eye(3));g=r.sides['Left'];height=(g['hull']@r.rot(g['foot'])[2]+r.point(g['foot'])[2]).min();rows[:,2]-=height;r.set(rows[0],np.eye(3))
    for side,g in r.sides.items():
        anchor=r.point(g['foot'],g['sole'])-np.array([.00097,0.,0.]);anchor[2]=0.
        t['sides'][side].update(contact=np.ones(n) if side=='Left' else np.zeros(n),pivots=np.tile(g['sole'],(n,1)),anchors=np.tile(anchor,(n,1)))
    geometry=Geometry(r,t,rows,0,n,np.array([0]),lambda b:True)
    z=rows[:,0].copy()
    assert geometry.evaluate(z+.000015)<1e-10
    assert geometry.evaluate(z+.00004)>0.



def test_collision_linearization_retains_every_near_dense_sample():
    from dex_frax.local_geometry import Geometry
    r=Robot();n=5;rows=np.tile(np.r_[0.,0.,2.,np.zeros(29)],(n,1))
    rows[:,3+r.names.index('hip_roll_l_joint')]=-.13
    rows[:,3+r.names.index('hip_roll_r_joint')]=.13
    t=source(n);t['body_contacts']=[]
    for side,g in r.sides.items():t['sides'][side].update(contact=np.zeros(n),pivots=np.tile(g['sole'],(n,1)),anchors=np.zeros((n,3)))
    ids=np.asarray(sum([g['leg'] for g in r.sides.values()],[]));geometry=Geometry(r,t,rows,0,n,ids,lambda b:True,restore=True)
    expected=sum(np.count_nonzero(sample[7]<.02) for sample in geometry.samples)
    unique={(sample[0],int(k)) for sample in geometry.samples for k in np.flatnonzero(sample[7]<.02)}
    A,lo,hi=geometry.evaluate(rows[:,ids].ravel(),linear=True)
    assert expected>len(unique) and len(lo)==expected
    assert np.any(np.sum(abs(A.toarray())>1e-12,axis=1)>1)
