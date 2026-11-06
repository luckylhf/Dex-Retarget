"""Source-specific anatomy, in a shared right-handed Z-up world frame."""
import json
from pathlib import Path
import numpy as np
from .bvh import load
from .source_profile import lafan_foot
from .math3d import unit, frame_yz, frame_xz

REFERENCE=Path(__file__).resolve().parent/'reference'

def adapt(motion, source='auto'):
    p,r=motion.fk();n=len(p['Hips']);fps=motion.fps
    if source=='auto':
        source='soma' if 'LeftShin' in p else ('mocap' if 'LeftHandIndex1' in p else 'lafan')
    if source not in ('soma','lafan','mocap'):raise ValueError(source)
    hipname,kneename=('Leg','Shin') if source=='soma' else ('UpLeg','Leg')
    neck='Neck1' if source=='soma' else 'Neck'
    spine=next(k for k in ('Spine','Spine1') if k in p)
    need=['Hips',neck,spine]+[s+k for s in ('Left','Right') for k in (hipname,kneename,'Foot','Arm','ForeArm','Hand')]
    if any(k not in p for k in need):raise ValueError('Missing anatomy: '+str([k for k in need if k not in p]))
    # LAFAN's first spine OFFSET is tilted ~20 deg despite its anatomical
    # pelvis-up axis being local +X (also the GMR G1 calibration). A bone's
    # slanted translation must not become a permanent pelvis rotation bias.
    pelvis_up=r['Hips']@np.array([0.,1.,0.] if source=='mocap' else [1.,0.,0.])
    root=frame_yz(p['Left'+hipname]-p['Right'+hipname],pelvis_up)
    torso=frame_yz(p['LeftArm']-p['RightArm'],p[neck]-p[spine])
    t={'root_r':root,'torso_r':torso,'hips':p['Hips'],'source_type':source,'fps':fps,'sides':{},'p':p,'r':r}
    offsets=dict(zip(motion.names,motion.offsets))
    if source=='soma':
        fp=REFERENCE/'NVIDIA/soma_retargeter/assets/soma/contact_processing/soma_contact_config.json'
        authored=json.loads(fp.read_text())['foot_landmarks']['rigs']['soma']
        zp,zr=load(REFERENCE/'soma_zero.bvh').fk()
        t['scene_ground']={'height_m':0.,'normal':[0,0,1],'method':'source_coordinate_convention','reference':'NVIDIA SOMA contact config ground plane z=0','measured':False}
    for s in ('Left','Right'):
        ankle=s+'Foot'; toe=next((k for k in (s+'ToeBase',s+'Toe',s+'Foot__EndSite') if k in p),None)
        if toe is None:raise ValueError('Foot needs toe marker or nonzero End Site: '+s)
        if source=='soma':
            a=authored[s.lower()]
            heel=p[ankle]+r[ankle]@np.array(a['heel_local_in_foot'])
            toe_p=p[a['toe_joint']]+r[a['toe_joint']]@np.array(a['toe_pivot_local_in_toe'])
            normal=unit(r[ankle]@np.array(a['sole_normal_local']))
            foot_frame=frame_xz(toe_p-heel,normal)
            foot_cal='NVIDIA authored SOMA heel/ball/sole normal; calibrated skeleton checked'
            # A fixed local forearm transverse reference from the official zero
            # pose transfers pronation, independently of animated wrist rotation.
            w0=zp[s+'HandIndex1'][0]-zp[s+'HandPinky1'][0]
            forearm_local=zr[s+'ForeArm'][0].T@w0
        else:
            raw=p[toe]-p[ankle]
            if source=='mocap':
                # This producer has local +Y up and +Z forward; End Site is
                # parsed from the file, never hard-coded to nobody's dimensions.
                if np.linalg.norm(offsets[toe])<.03:raise ValueError('Unusable mocap Foot End Site')
                forward=np.array([0.,0.,1.]);up=np.array([0.,1.,0.])
                foot_frame=frame_xz(r[ankle]@forward,r[ankle]@up)
                forearm_local=offsets[s+'HandIndex1']-offsets[s+'HandPinky1']
                foot_cal='mocap Y-up foot frame; parsed End Site toe; inferred heel'
            else:
                foot_frame,heel,toe_p,foot_cal,scene=lafan_foot(motion,p,r,s)
                t['scene_ground']=scene
                forearm_local=np.array([0.,-1.,0.])
            if source=='mocap':
                toe_p=p[toe]
                depth=np.median(np.einsum('ni,ni->n',p[ankle]-toe_p,foot_frame[:,:,2]))
                back=.25*float(np.median(np.linalg.norm(raw,axis=1)))
                heel=p[ankle]-foot_frame[:,:,0]*back-foot_frame[:,:,2]*depth
        upper=unit(p[s+'ForeArm']-p[s+'Arm']);lower=unit(p[s+'Hand']-p[s+'ForeArm'])
        # Calibrated anatomical flexion axes in the UPPER-arm bone frame.
        # Preserve the authored Arm rotation even when three bone points become
        # almost collinear. A per-frame cross product cannot resolve this case.
        local_axis=np.array([0.,0.,1.] if source=='soma' else
            ([0.,-1. if s=='Left' else 1.,0.] if source=='mocap' else [0.,0.,-1.]))
        elbow_axis=r[s+'Arm']@local_axis
        elbow_axis=unit(elbow_axis-upper*np.sum(elbow_axis*upper,axis=1,keepdims=True))
        dot=np.clip(np.sum(upper*lower,axis=1),-1.,1.)
        signed_bend=np.arctan2(np.sum(elbow_axis*np.cross(upper,lower),axis=1),dot)
        angle=np.arccos(dot);cal_frames=(angle>np.deg2rad(25))&(angle<np.deg2rad(140))
        normal=unit(np.cross(upper,lower))
        errors=np.degrees(np.arccos(np.clip(np.sum(normal[cal_frames]*elbow_axis[cal_frames],axis=1),-1,1)))
        elbow_calibration={'method':'source-profile upper-arm flexion axis; signed bend',
            'local_axis':local_axis.tolist(),'check_frames':int(cal_frames.sum()),
            'axis_residual_median_deg':float(np.median(errors)) if errors.size else None,
            'axis_residual_p95_deg':float(np.percentile(errors,95)) if errors.size else None}
        if s+'HandIndex1' in p:
            forward=unit(p[s+'HandMiddle1']-p[s+'Hand'])
            width=p[s+'HandIndex1']-p[s+'HandPinky1']
            width=unit(width-(width*forward).sum(axis=1,keepdims=True)*forward)
            hand_method='finger bases: middle direction and index-minus-pinky width'
        else:
            forward=unit(r[s+'Hand']@unit(offsets[s+'Hand']))
            width=unit(r[s+'Hand']@np.array([0.,-1.,0.]))
            width=unit(width-(width*forward).sum(axis=1,keepdims=True)*forward)
            hand_method='LAFAN hand +X distal/-Y width, anatomical calibration from GMR G1'
        fw=r[s+'ForeArm']@unit(forearm_local)
        fw=unit(fw-(fw*lower).sum(axis=1,keepdims=True)*lower)
        t['sides'][s]={'hip':p[s+hipname],'knee':p[s+kneename],'ankle':p[ankle],
            'thigh':unit(p[s+kneename]-p[s+hipname]),'shin':unit(p[ankle]-p[s+kneename]),
            'upper':upper,'lower':lower,'elbow_axis':elbow_axis,'signed_elbow_bend':signed_bend,
            'elbow_calibration':elbow_calibration,
            'hand_forward':forward,'hand_width':width,'forearm_width':fw,
            'foot_r':foot_frame,'heel':heel,'toe':toe_p,'foot_calibration':foot_cal,'hand_method':hand_method}
    return t
