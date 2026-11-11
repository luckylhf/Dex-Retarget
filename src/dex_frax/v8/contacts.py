"""Source contact inference and 3D target correction.

Mathematical adaptation of NVIDIA v0.2.1's coarse-contact -> plant gates ->
median anchors -> smootherstep propagation. Extensions: toe/heel pivots,
physical derivative units, time-based gates, and explicit ground height.
All labels are estimates, never force/pressure measurements.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from .math3d import runs,smoother,frame_xz,blend_rot

def estimate_ground(t,cfg):
    """Prefer the low envelope unless stable flat feet contradict its height.

    A few moving/penetrating mocap markers are not evidence for a lower floor.
    Require persistent flat, slow observations from BOTH feet before rejecting
    that envelope. This is still an inferred single plane, not force sensing.
    """
    scene=cfg.get('scene_ground_override') or t.get('scene_ground')
    if scene is not None:
        height=float(scene['height_m'])
        if not np.isfinite(height) or not np.allclose(scene.get('normal',[0,0,1]),[0,0,1]):
            raise ValueError('Ground scene must be a finite horizontal plane in source Z-up coordinates')
        return height,{**scene,'method':scene['method'],'inferred_single_plane':not scene.get('measured',False),'foot_height_does_not_set_floor':True}
    heights=np.concatenate([v[k][:,2] for v in t['sides'].values() for k in ('heel','toe')])
    low=float(np.percentile(heights,2));candidates=[];counts={}
    for side,v in t['sides'].items():
        center=(v['heel']+v['toe'])/2
        sm=gaussian_filter1d(center,max(.5,.018*t['fps']),axis=0,mode='nearest')
        speed=np.linalg.norm(np.gradient(sm,1/t['fps'],axis=0),axis=1)
        stable=(speed<cfg['speed_enter'])&(v['foot_r'][:,2,2]>np.cos(np.deg2rad(cfg['flatness_deg'])))
        stable &= abs(v['heel'][:,2]-v['toe'][:,2])<cfg['height_spread_m']
        for a,b in runs(stable):
            if b-a<max(3,round(cfg['stable_duration_s']*t['fps'])):stable[a:b]=False
        counts[side]=int(stable.sum())
        candidates.extend(np.minimum(v['heel'][:,2],v['toe'][:,2])[stable])
    ground=low;method='low_envelope'
    robust=float(np.percentile(candidates,25)) if candidates else None
    if robust is not None and min(counts.values())>=max(3,round(.25*t['fps'])):
        if robust-low>cfg['height_tolerance_m']*.5:
            ground=robust;method='persistent_flat_support_override'
    return ground,{'method':method,'low_envelope_m':low,'flat_support_lower_quartile_m':robust,
                   'flat_observation_counts':counts,'inferred_single_plane':True}

def stable_mask(points,fps,ground,cfg):
    # Correct physical units; NVIDIA's historical v/(2dt) thresholds must not
    # be copied numerically. Smoothing derivatives helps BVH quantization.
    sm=gaussian_filter1d(points,max(.5,.018*fps),axis=0,mode='nearest')
    v=np.gradient(sm,1/fps,axis=0);speed=np.linalg.norm(v,axis=1)
    acc=np.linalg.norm(np.gradient(v,1/fps,axis=0),axis=1)
    jerk=np.gradient(acc,1/fps)
    low=points[:,2]<ground+cfg['height_tolerance_m']
    mask=np.zeros(len(points),bool);active=False;window=0
    for i in range(len(points)):
        if jerk[i]<0:window=max(1,round(fps*cfg['jerk_window_s']))
        else:window=max(0,window-1)
        # A stationary segment at either clip boundary has no jerk event.
        enter=speed[i]<cfg['speed_enter'] and (window>0 or speed[i]<.03)
        if not active and low[i] and enter:active=True
        if active and (not low[i] or speed[i]>cfg['speed_leave']):active=False
        mask[i]=active
    # Conservative plant gate following the coarse state machine.
    mask &= speed<cfg['speed_enter']
    for a,b in runs(mask):
        if b-a<max(3,round(cfg['stable_duration_s']*fps)):mask[a:b]=False
    return mask,speed,jerk

def prepare(t,robot,cfg,enabled=True):
    fps=t['fps'];n=len(t['hips'])
    scale=float(np.mean([(robot.sides[s]['thigh_length']+robot.sides[s]['shin_length'])/
        np.median(np.linalg.norm(v['knee']-v['hip'],axis=1)+np.linalg.norm(v['ankle']-v['knee'],axis=1))
        for s,v in t['sides'].items()]))
    ground,calibration=estimate_ground(t,cfg)
    origin=np.r_[t['hips'][0,:2],ground]
    t['root_p']=(t['hips']-origin)*scale
    t['scale']=scale;t['floor']=ground;t['world_origin']=origin
    t['ground_calibration']=calibration
    v2 = cfg.get('policy') == 'v2'
    minimum_frames = max(3, round(cfg['stable_duration_s'] * fps))
    if v2:
        from .foot_contact_v2 import orientation_targets, support_labels
    for s,v in t['sides'].items():
        g=robot.sides[s];r=v['foot_r'].copy()
        hm,hs,hj=stable_mask(v['heel'],fps,ground,cfg);tm,ts,tj=stable_mask(v['toe'],fps,ground,cfg)
        tilt=np.degrees(np.arccos(np.clip(r[:,2,2],-1,1)))
        spread_limit=cfg['height_spread_m']
        if t['source_type']=='lafan':
            # SOMA's 5cm gate refers to its shorter authored landmarks. Using
            # it unchanged across the full reference shoe would impose a much
            # stricter angle and falsely turn quiet flat plants into toe plants.
            span=np.linalg.norm(v['heel']-v['toe'],axis=1)
            spread_limit=np.maximum(spread_limit,span*np.sin(np.deg2rad(cfg['flatness_deg'])))
        flat=hm&tm&(tilt<cfg['flatness_deg'])&(np.abs(v['heel'][:,2]-v['toe'][:,2])<spread_limit)
        labels=np.zeros(n,np.int8)
        labels[tm&~flat]=2;labels[hm&~flat&~tm]=3;labels[flat]=1
        if v2:
            labels = support_labels(v['heel'], v['toe'], hm, tm, minimum_frames, cfg['v2'])
        # Foot center scales in world XYZ. Robot ankle offset comes from its
        # actual sole, not a copied human ankle height or ankle-to-toe pitch.
        center=((v['heel']+v['toe'])/2-origin)*scale
        pos=center-np.einsum('nij,j->ni',r,g['sole'])
        desired_low=np.maximum(0.,(np.minimum(v['heel'][:,2],v['toe'][:,2])-ground)*scale)
        minlocal=np.min(np.einsum('nij,kj->nki',r,g['hull'])[:,:,2],axis=1)
        pos[:,2]=desired_low-minlocal
        rawpos=pos.copy();rawr=r.copy();weights=np.zeros(n);anchors=np.zeros((n,3));pivots=np.zeros((n,3));segments=[]
        if not enabled:labels[:]=0
        # Keep only intervals with a stable source pivot. Select all accepted
        # intervals before constructing v2's continuous correction envelope.
        accepted = []
        for label, pivot_name in [(1, 'sole'), (2, 'toe'), (3, 'heel')]:
            src = (v['heel'] + v['toe']) / 2 if label == 1 else v[pivot_name]
            for a, b in runs(labels == label):
                if b - a < minimum_frames:
                    labels[a:b] = 0
                    continue
                median = np.median(src[a:b], axis=0)
                excursion = np.max(np.linalg.norm(src[a:b] - median, axis=1))
                if excursion > cfg['max_anchor_excursion_m']:
                    labels[a:b] = 0
                    continue
                accepted.append((a, b, label, pivot_name, excursion))
        free_base = rawpos
        if v2:
            r, flat_confidence, side_evidence, roll_level = orientation_targets(
                rawr, labels, v, hm, tm, hs, ts, ground, fps, cfg,
            )
            v.update(foot_flat_confidence=flat_confidence, foot_side_evidence=side_evidence,
                     foot_roll_level_weight=roll_level)
            pos = center - np.einsum('nij,j->ni', r, g['sole'])
            pos[:, 2] = desired_low - np.min(r[:, 2, :] @ g['hull'].T, axis=1)
            free_base = pos.copy()
        for a, b, label, pivot_name, excursion in accepted:
            local=np.broadcast_to(g[pivot_name],(b-a,3)).copy()
            rawpivot=rawpos[a:b]+np.einsum('nij,nj->ni',rawr[a:b],local)
            anchor=np.median(rawpivot,axis=0);anchor[2]=0.
            target_r = r[a:b].copy() if v2 else rawr[a:b].copy()
            if not v2 and label==1:
                avg=Rotation.from_matrix(target_r).mean().as_matrix()
                target_r[:]=frame_xz(avg[:,0],np.array([0.,0.,1.]))
            # Anchor the semantic pivot in XY and align the actual shoe
            # minimum to the floor, using the corrected orientation.
            target_p=anchor-np.einsum('nij,nj->ni',target_r,local)
            target_p[:,2]=-np.min(target_r[:,2,:]@g['hull'].T,axis=1)
            pos[a:b]=target_p;r[a:b]=target_r
            weights[a:b]=1.;anchors[a:b]=anchor;pivots[a:b]=local
            segments.append({'start':int(a),'end':int(b),'phase':pivot_name,'label':label,
                'anchor_m':anchor.tolist(),'source_excursion_m':float(excursion)})
        segments.sort(key=lambda s:s['start'])
        # If support changes type without a swing interval, carry the old
        # foot's placement into the new pivot, rather than invent a new step.
        for previous,current in zip(segments[:-1],segments[1:]):
            a,b=current['start'],current['end'];end=previous['end']
            if end==a:
                anchor=pos[a-1]+r[a-1]@pivots[a]
                anchor[2]=0.
                anchors[a:b]=anchor
                pos[a:b]=anchor-np.einsum('nij,nj->ni',r[a:b],pivots[a:b])
                pos[a:b,2]=-np.min(r[a:b,2,:]@g['hull'].T,axis=1)
                current['anchor_m']=anchor.tolist()
                current['continuous_from_previous']=True
        # Decay the boundary residual into FREE frames only. Multiple nearby
        # anchors blend by normalized weights; never overwrite another plant.
        corrected=labels>0;delta=np.zeros((n,3));rsum=np.zeros((n,3));wsum=np.zeros(n)
        ramp=max(1,round(fps*cfg['transition_s']))
        for seg in segments:
            a,b=seg['start'],seg['end']
            for edge,sgn in [(a,-1),(b-1,1)]:
                dp = pos[edge] - free_base[edge]
                dr = np.zeros(3) if v2 else Rotation.from_matrix(rawr[edge].T@r[edge]).as_rotvec()
                for k in range(1,ramp+1):
                    j=edge+sgn*k
                    if j<0 or j>=n or corrected[j]:break
                    w=smoother(1-k/(ramp+1));delta[j]+=w*dp;rsum[j]+=w*dr;wsum[j]+=w
        den=np.maximum(1.,wsum)
        pos[~corrected]+=delta[~corrected]/den[~corrected,None]
        if v2:
            minimum = -np.min(r[~corrected, 2, :] @ g['hull'].T, axis=1)
            pos[~corrected, 2] = np.maximum(pos[~corrected, 2], minimum)
        else:
            r[~corrected]=rawr[~corrected]@Rotation.from_rotvec(rsum[~corrected]/den[~corrected,None]).as_matrix()
        v.update(foot_target_p=pos,foot_target_r=r,contact=weights,phase=labels,anchors=anchors,pivots=pivots,
                 segments=segments,raw_foot_p=rawpos,raw_foot_r=rawr,heel_speed=hs,toe_speed=ts)
    from .multicontact import prepare as prepare_body_contacts
    return prepare_body_contacts(t,robot,cfg,enabled)
