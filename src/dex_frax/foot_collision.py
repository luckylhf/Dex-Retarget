"""Full feet against the other leg; external foot regions against own shin.

The ankle mounting corridor is derived from model geometry, not a pose learned
from an arbitrary motion. The original robot/ground-contact model is untouched.
"""
from collections import OrderedDict
import copy
import numpy as np
import mujoco
from scipy.spatial import ConvexHull

POLICY='full_feet_and_external_foot_shin_v1'
_CACHE=OrderedDict()


def vertices(robot,geom,body):
    m=robot.m;mid=int(m.geom_dataid[geom])
    v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
    world=v@robot.d.geom_xmat[geom].reshape(3,3).T+robot.d.geom_xpos[geom]
    return (world-robot.point(body))@robot.rot(body)


def clipped_hull(points,axis,bound,sign):
    hull=ConvexHull(points);points=np.asarray(points);distance=sign*(points[:,axis]-bound)
    rows=list(points[distance>=0])
    edges={tuple(sorted((int(face[a]),int(face[b])))) for face in hull.simplices for a,b in ((0,1),(1,2),(2,0))}
    for a,b in edges:
        if distance[a]*distance[b]<0:
            u=distance[a]/(distance[a]-distance[b]);rows.append(points[a]+u*(points[b]-points[a]))
    result=np.unique(np.round(rows,12),axis=0)
    if len(result)<4 or np.linalg.matrix_rank(result-result.mean(0))<3:return None
    h=ConvexHull(result)
    return result[h.vertices] if h.volume>1e-12 else None


def augmented(robot):
    key=id(robot.m)
    if key in _CACHE:
        _CACHE.move_to_end(key);_,model,record=_CACHE[key]
        return copy.copy(model),copy.deepcopy(record)
    saved=robot.d.qpos.copy();spec=mujoco.MjSpec.from_file(robot.path);record={'policy':POLICY,'sides':{}}
    try:
        robot.set(np.r_[0.,0.,1.,np.zeros(29)],np.eye(3))
        for side,suffix in (('Left','l'),('Right','r')):
            g=robot.sides[side];foot=g['foot'];name='ankle_roll_'+suffix+'_link'
            v=vertices(robot,robot.m.geom(name+'_visual').id,foot)
            mount=vertices(robot,robot.m.geom('ankle_pitch_'+suffix+'_link_visual').id,foot)
            shin=vertices(robot,robot.m.geom('knee_pitch_'+suffix+'_link_collision').id,foot)
            near=clipped_hull(shin,2,float(v[:,2].max()+1e-6),-1)
            if near is None:raise ValueError('Cannot identify ankle mounting geometry')
            mount=np.vstack([mount,near]);lo=mount.min(0);hi=mount.max(0)
            # The distal shin mesh includes the ankle socket. Keep that joint
            # in the original collision model; extra foot-volume checks use
            # the shaft above the ankle hardware's geometric radius.
            hardware=vertices(robot,robot.m.geom('ankle_pitch_'+suffix+'_link_collision').id,foot)
            shaft_start=float(np.max(np.linalg.norm(hardware,axis=1)))
            shaft=clipped_hull(shin,2,shaft_start,1)
            if shaft is None:raise ValueError('Missing shin shaft geometry')
            # Mesh vertices must be expressed in the owning shin frame.
            world=shaft@robot.rot(foot).T+robot.point(foot)
            shaft_local=(world-robot.point(g['knee']))@robot.rot(g['knee'])
            shaft_name='retarget_shin_'+suffix+'_shaft'
            spec.add_mesh(name=shaft_name,uservert=shaft_local.flatten().tolist())
            spec.body('knee_pitch_'+suffix+'_link').add_geom(name=shaft_name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=shaft_name,group=3,contype=1,conaffinity=1,density=0.)
            parts=[]
            for label,axis,bound,sign in [('heel',0,lo[0],-1),('toe',0,hi[0],1),('plate',2,lo[2],-1)]:
                points=clipped_hull(v,axis,float(bound),sign)
                if points is None:continue
                part='retarget_foot_'+suffix+'_'+label
                spec.add_mesh(name=part,uservert=points.flatten().tolist())
                spec.body(name).add_geom(name=part,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=part,group=3,contype=1,conaffinity=1,density=0.)
                parts.append(part)
            record['sides'][side]=dict(full_geom=name+'_visual',shin_geom='knee_pitch_'+suffix+'_link_collision',shaft_geom=shaft_name,shaft_start_m=shaft_start,external_parts=parts,
                joint_corridor=dict(x=[float(lo[0]),float(hi[0])],y=[float(lo[1]),float(hi[1])],z_min=float(lo[2])))
        model=spec.compile()
        # Geometry additions may renumber geoms, never bodies or coordinates.
        if model.nq!=robot.m.nq or model.nv!=robot.m.nv or model.nbody!=robot.m.nbody:raise ValueError('Foot collision model changed kinematics')
        for i in range(model.nbody):
            if model.body(i).name!=robot.m.body(i).name:raise ValueError('Foot collision body mapping changed')
        if not np.array_equal(model.jnt_qposadr,robot.m.jnt_qposadr):raise ValueError('Foot collision joint mapping changed')
    finally:
        from scipy.spatial.transform import Rotation
        robot.set(np.r_[saved[:3],saved[robot.qadr]],Rotation.from_quat(saved[3:7][[1,2,3,0]]).as_matrix())
    _CACHE[key]=(robot.m,model,record)
    while len(_CACHE)>4:_CACHE.popitem(last=False)
    return copy.copy(model),copy.deepcopy(record)


def groups(model,record,ids):
    full_ids=[model.geom(v['full_geom']).id for v in record['sides'].values()]
    pairs=[]
    for side,v in record['sides'].items():
        foot=model.geom(v['full_geom']).id;shin=model.geom(v['shin_geom']).id
        for other in list(ids)+full_ids:
            if other not in (foot,shin):pairs.append(([foot],[int(other)]))
        # Do not omit the whole same-leg foot/shin pair due to its joint mount.
        for part in v['external_parts']:pairs.append(([model.geom(part).id],[model.geom(v['shaft_geom']).id]))
    return pairs,full_ids+[model.geom(name).id for v in record['sides'].values() for name in v['external_parts']+[v['shaft_geom']]]
