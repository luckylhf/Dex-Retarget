"""Source-relative leg-shape protection during a geometry repair."""
import numpy as np


def errors(robot,t,rows,a,b):
    result=[]
    for i in range(a,b):
        robot.set(rows[i],t['root_r'][i]);values=[]
        for side,g in robot.sides.items():
            for proximal,distal,key in [('hip','knee','thigh'),('knee','foot','shin')]:
                d=robot.point(g[distal])-robot.point(g[proximal]);d/=max(np.linalg.norm(d),1e-12)
                values.append(np.degrees(np.arccos(np.clip(d@t['sides'][side][key][i],-1.,1.))))
        result.append(values)
    return np.asarray(result)


def tradeoff(before,after,old_deficit,new_deficit):
    """Safety first, shape soft: reject a dominated candidate, not all tradeoffs."""
    before=np.asarray(before,float);after=np.asarray(after,float)
    old=float(np.mean(before**2));new=float(np.mean(after**2))
    safety_gain=new_deficit<old_deficit-1e-7
    worsened=new>old+1e-7
    return dict(before_rms_deg=float(np.sqrt(old)),after_rms_deg=float(np.sqrt(new)),
        before_p95_deg=np.percentile(before,95,axis=0).tolist(),after_p95_deg=np.percentile(after,95,axis=0).tolist(),
        direction_order=['Left_thigh','Left_shin','Right_thigh','Right_shin'],
        owned_collision_deficit_before=old_deficit,owned_collision_deficit_after=new_deficit,
        shape_cost_for_safety=bool(worsened and safety_gain),dominated=bool(worsened and not safety_gain),
        policy='reject shape regression without owned collision improvement; no absolute leg similarity certification')
