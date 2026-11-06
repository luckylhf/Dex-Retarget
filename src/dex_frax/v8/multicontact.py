"""Scene-plane contacts, including free flight, fixed plants and moving contacts.

Human BVHs have no force/skin measurements. Non-foot contact candidates use
robot-surface proxies aligned to source anatomical frames; this uncertainty is
recorded. A candidate is not a measured load-bearing contact. Sliding/rolling
contacts constrain the normal only and are never turned into fixed XYZ plants.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from .math3d import frame_xz,frame_yz,runs,smoother

_SURFACE_CACHE={}


def surfaces(robot):
    cache=_SURFACE_CACHE
    if cache.get('model') is robot.m:return cache['surfaces']
    m,d=robot.m,robot.d;groups={}
    for gid in range(m.ngeom):
        mid=int(m.geom_dataid[gid]);body=int(m.geom_bodyid[gid])
        if mid<0 or int(m.geom_group[gid])!=2:continue
        vertices=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        world=vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
        local=(world-d.xpos[body])@robot.rot(body)
        groups.setdefault(body,[]).append(local)
    result={}
    for body,parts in groups.items():
        v=np.concatenate(parts);hull=v[ConvexHull(v).vertices]
        result[body]=dict(body=body,name=m.body(body).name,hull=hull)
    cache.clear();cache.update(model=m,surfaces=result)
    return result


def lowest(robot,body):
    v=robot.surface_models[body]['hull']
    return v[np.argmin(v@robot.rot(body)[2])]


def region_hull(robot,key,body):
    """Union of the actual fixed component surfaces in a semantic region."""
    shape=robot.surface_models
    names=([g['name'] for g in shape.values() if 'head' in g['name']] if key=='head' else
           ['waist_pitch_link','camera_body_front_link','imu_waist_link'] if key=='torso' else
           [shape[body]['name'],'wrist_roll_'+('l' if key.startswith('Left') else 'r')+'_link'] if key.endswith('_hand') else [shape[body]['name']])
    chunks=[]
    for other,g in shape.items():
        if g['name'] not in names or robot.m.body_weldid[other]!=robot.m.body_weldid[body]:continue
        world=g['hull']@robot.rot(other).T+robot.point(other)
        chunks.append((world-robot.point(body))@robot.rot(body))
    return np.concatenate(chunks) if chunks else shape[body]['hull']


def contact_local(robot,contact):
    """Actual support vertex of a semantic region containing fixed parts."""
    v=np.asarray(contact.get('robot_hull',robot.surface_models[contact['body']]['hull']))
    return v[np.argmin(v@robot.rot(contact['body'])[2,:])]


def classify_contact(points,rotations,height,fps,cfg):
    """0 free; 1 fixed material point; 2 sliding; 3 rolling/changing surface."""
    points=np.asarray(points);rotations=np.asarray(rotations);height=np.asarray(height)
    normal_speed=np.abs(np.gradient(height,1/fps))
    point_speed=np.linalg.norm(np.gradient(points,1/fps,axis=0),axis=1)
    angular=np.r_[0.,Rotation.from_matrix(rotations[1:]@rotations[:-1].transpose(0,2,1)).magnitude()*fps]
    candidate=(abs(height)<cfg['height_tolerance_m'])&(normal_speed<cfg['speed_leave'])
    minimum=max(3,round(cfg['stable_duration_s']*fps))
    for a,b in runs(candidate):
        if b-a<minimum:candidate[a:b]=False
    mode=np.zeros(len(height),np.int8)
    mode[candidate]=2
    mode[candidate&(point_speed<cfg['speed_enter'])&(angular<cfg.get('fixed_contact_angular_speed_rad_s',.5))]=1
    mode[candidate&(angular>=cfg.get('fixed_contact_angular_speed_rad_s',.5))]=3
    return mode


def prepare(t,robot,cfg,enabled=True):
    """Infer additional contacts without changing the calibrated rig or floor."""
    n=len(t['hips']);fps=t['fps'];p=t['p'];scale=t['scale'];origin=t['world_origin']
    # Geometry was cached at Robot's neutral configuration. Reconstruct it now
    # before calculating source proxies, independent of the current frame.
    robot.set(np.r_[0.,0.,1.,np.zeros(29)],np.eye(3))
    shape=robot.surface_models;models=[]
    source_surface=None
    if t['source_type'] in ('lafan','soma'):
        filename='lafan1_surfaces.json' if t['source_type']=='lafan' else 'soma_surfaces.json'
        source_surface=json.loads(Path(__file__).with_name('reference').joinpath(filename).read_text())
    def add(key,body,source_pos,source_rot,neutral_origin,neutral_rot):
        if body not in shape:return
        v=shape[body]['hull'];world=v@robot.rot(body).T+robot.point(body)
        proxy=(world-neutral_origin)@neutral_rot/scale
        geometry_status='target visual convex envelope aligned to source anatomy; inferred, not measured'
        if source_surface is not None:
            bone=source_surface['semantic_bones'][key]
            proxy=np.array(source_surface['bones'][bone]['vertices'])
            source_pos=t['p'][bone];source_rot=t['r'][bone]
            geometry_status='official source reference skin, dominant-binding rigid bone hull; inferred, not measured'
        # Candidate height is the supporting surface, not the bone center.
        z=np.einsum('nj,kj->nk',source_rot[:,2,:],proxy)+source_pos[:,2,None]
        idx=np.argmin(z,axis=1);local=proxy[idx]
        point=source_pos+np.einsum('nij,nj->ni',source_rot,local)
        height=point[:,2]-t['floor']
        mode=classify_contact(point,source_rot,height,fps,cfg)
        if not enabled:mode[:]=0
        target=(point-origin)*scale
        # Only true fixed-material segments get an XY anchor. Moving/rolling
        # modes retain changing XY and will receive only a normal constraint.
        anchors=target.copy();anchors[:,2]=0.
        for a,b in runs(mode==1):
            if b-a<max(3,round(cfg['stable_duration_s']*fps)):
                mode[a:b]=2;continue
            if np.max(np.linalg.norm(point[a:b]-np.median(point[a:b],axis=0),axis=1))>cfg['max_anchor_excursion_m']:
                mode[a:b]=2;continue
            anchors[a:b,:2]=np.median(target[a:b,:2],axis=0)
        region=region_hull(robot,key,body)
        confidence=np.zeros(n);ramp=max(1,round(fps*cfg['transition_s']))
        for a,b in runs(mode>0):
            distance=np.minimum(np.arange(1,b-a+1),np.arange(b-a,0,-1))
            confidence[a:b]=smoother(np.clip(distance/ramp,0.,1.))
        models.append(dict(key=key,body=body,body_name=shape[body]['name'],robot_hull=region.tolist(),mode=mode,confidence=confidence,anchors=anchors,source_surface_height=height,source_surface_point=point,proxy_geometry=geometry_status,source_geometry_confidence=1. if source_surface is not None else .25))
    add('pelvis',robot.bid('pelvis'),p['Hips'],t['root_r'],robot.point(robot.bid('pelvis')),robot.rot(robot.bid('pelvis')))
    center=(p['LeftArm']+p['RightArm'])/2
    rc=(robot.point(robot.sides['Left']['shoulder'])+robot.point(robot.sides['Right']['shoulder']))/2
    torso_origin=robot.point(robot.torso);offset=(torso_origin-rc)@robot.rot(robot.torso)
    center=center+np.einsum('nij,j->ni',t['torso_r'],offset/scale)
    add('torso',robot.torso,center,t['torso_r'],torso_origin,robot.rot(robot.torso))
    head=robot.bid('head_yaw_link')
    if 'Head' in p:add('head',head,p['Head'],t['torso_r'],robot.point(head),robot.rot(head))
    for side,g in robot.sides.items():
        v=t['sides'][side]
        for key,b,src,vector in [('thigh',robot.bid('hip_yaw_'+('l' if side=='Left' else 'r')+'_link'),v['hip'],v['thigh']),('shin',g['knee'],v['knee'],v['shin'])]:
            # Source and target use the same semantic longitudinal/lateral axes.
            if key=='thigh':neutral=robot.point(g['knee'])-robot.point(g['hip'])
            else:neutral=robot.point(g['foot'])-robot.point(g['knee'])
            nr=frame_yz(np.array([0.,1.,0.]),-neutral)
            sr=frame_yz(t['root_r'][:,:,1],-vector)
            add(side+'_'+key,b,src,sr,robot.point(g['hip'] if key=='thigh' else g['knee']),nr)
        for key,b,src,vector in [('upper_arm',g['shoulder'],p[side+'Arm'],v['upper']),('forearm',g['forearm'],p[side+'ForeArm'],v['lower'])]:
            if key=='upper_arm':
                neutral=robot.point(g['elbow'])-robot.point(g['shoulder']);nr=frame_yz(robot.rot(g['shoulder'])[:,1],-neutral);sr=frame_yz(-v['elbow_axis'],-vector)
            else:
                neutral=robot.point(g['wrist'])-robot.point(g['elbow']);nr=frame_xz(robot.rot(g['forearm'])[:,0],-neutral);sr=frame_xz(v['forearm_width'],-vector)
            add(side+'_'+key,b,src,sr,robot.point(g['shoulder'] if key=='upper_arm' else g['elbow']),nr)
        sr=frame_xz(v['hand_width'],-v['hand_forward'])
        add(side+'_hand',g['tcp'],p[side+'Hand'],sr,robot.point(g['wrist']),robot.rot(g['tcp']))
    feet=np.minimum.reduce([v[k][:,2]-t['floor'] for v in t['sides'].values() for k in ['heel','toe']])
    expected=np.logical_or.reduce([v['contact']>0 for v in t['sides'].values()]+[v['mode']>0 for v in models])
    minimum=np.minimum.reduce([feet]+[v['source_surface_height'] for v in models])
    t['body_contacts']=models;t['source_expected_contact']=expected
    t['body_contact_height_tolerance_m']=float(cfg['height_tolerance_m']*scale)
    for side,v in t['sides'].items():
        v['ground_contact_confidence']=np.maximum.reduce([np.zeros(n)]+[c['confidence']*c['source_geometry_confidence'] for c in models if c['key'] in [side+'_hand',side+'_forearm',side+'_upper_arm']])
    t['source_airborne']=(minimum>cfg['height_tolerance_m'])&~expected
    t['multicontact_record']=dict(method='scene_plane_surface_candidates_v1',geometry_status='official reference skin for LAFAN/SOMA; explicitly lower-confidence robot morphology proxies for other BVHs; never measured forces',mode_names={'0':'free','1':'fixed','2':'sliding','3':'rolling'},counts={v['key']:{str(k):int(np.sum(v['mode']==k)) for k in [1,2,3]} for v in models},airborne_frames=int(t['source_airborne'].sum()))
    return t
