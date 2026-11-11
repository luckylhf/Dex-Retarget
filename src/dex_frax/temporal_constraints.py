"""Linear source-supported temporal corridors for the existing local fit.

These are optimization constraints derived from the unchanged quality screen,
not motor acceleration limits. Existing non-target motion is a feasible start;
flagged impulses must leave the screen's decision region with a small interior.
"""
import numpy as np
from scipy import sparse
from .temporal_quality import assess, source_steps, region, POLICY


def diagnostics(targets):
    out={'root_r':targets['root_r'],'torso_target':targets.get('torso_r',targets['root_r'])}
    for side,values in targets.get('sides',{}).items():
        for key in ('upper','lower','hand_forward','hand_width','elbow_axis','thigh','shin','foot_target_r'):
            if key in values:out[side+'_'+key]=values[key]
    return out


def corridor(robot,targets,reference,start,end,ids):
    ids=np.asarray(ids);n=end-start;d=len(ids);dg=diagnostics(targets)
    screen=assess(reference[:,3:],targets['fps'],robot.names,dg)
    sources={};rows=[];cols=[];values=[];limits=[];owned=[]
    degree=np.pi/180.
    for column,j in enumerate(ids):
        if j<3:continue
        name=robot.names[j-3];area=region(name)
        if area not in sources:sources[area]=source_steps(dg,area,len(reference))
        source=sources[area]
        if source is None:continue
        src=np.radians(source);flagged=set(screen['joints'][name]['violating_frames'])
        for order,coefficients in ((2,(1.,-2.,1.)),(3,(.5,-1.5,1.5,-.5))):
            for k in range(n-order):
                absolute=start+k;local_source=float(np.max(src[absolute:absolute+order]))
                base=float(np.dot(coefficients,reference[absolute:absolute+order+1,j]))
                # An isolated velocity pulse is tested at its central edge;
                # a sustained reversal screen marks its surrounding samples.
                affected=any(i in flagged for i in range(absolute,absolute+order+1))
                target=(1.8 if order==2 else .9)*max(POLICY['minimum_step_deg']*degree,
                    local_source/POLICY['source_step_ratio'])
                bound=target if affected else max(target,abs(base)+1e-8)
                row=len(limits)
                for h,value in enumerate(coefficients):rows.append(row);cols.append((k+h)*d+column);values.append(value)
                limits.append(bound);owned.append(affected)
    A=sparse.csc_matrix((values,(rows,cols)),shape=(len(limits),n*d))
    return A,np.asarray(limits),dict(policy='source_supported_temporal_corridor_v1',rows=len(limits),
        targeted_rows=int(sum(owned)),hardware_acceleration_limit=False)


def regressions(before,after):
    """A different joint's new defect cannot hide under an old union frame."""
    result={}
    for name,value in after.get('joints',{}).items():
        added=set(value.get('violating_frames',[]))-set(before.get('joints',{}).get(name,{}).get('violating_frames',[]))
        if added:result[name]=sorted(added)
    return result
