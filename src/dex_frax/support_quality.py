"""Separate reliable inferred support from uncertain geometric contact hints."""
import numpy as np

POLICY=dict(name='confidence_aware_body_support_v1',confidence_min=.8,
            tolerance_basis='existing source contact height/anchor tolerances scaled to robot metres',
            measured_load=False)


def evaluate_contacts(meta,dg,errors,n,fps):
    cfg=meta.get('configuration',{}).get('contact',{})
    scale=float(meta.get('leg_scale',1.))
    normal_limit=float(cfg.get('height_tolerance_m',.035))*scale
    xy_limit=float(cfg.get('max_anchor_excursion_m',.035))*scale
    bad=np.zeros(n,bool);report={}
    for c in meta.get('body_contact_models',[]):
        name=c['body_name'];prefix='body_'+name+'_'
        mode=np.asarray(dg[prefix+'mode']);confidence=np.asarray(dg.get(prefix+'confidence',np.zeros(n)))
        reliable=(mode>0)&(confidence*float(c.get('source_geometry_confidence',0.))>=POLICY['confidence_min'])
        normal=np.array([np.nan if v is None else v/100 for v in errors[name]['normal_cm']])
        xy=np.array([np.nan if v is None else v/100 for v in errors[name]['fixed_xy_cm']])
        normal_bad=reliable&((~np.isfinite(normal))|(normal>normal_limit))
        xy_bad=reliable&(mode==1)&((~np.isfinite(xy))|(xy>xy_limit))
        bad|=normal_bad|xy_bad
        report[name]=dict(trusted_frames=int(reliable.sum()),hint_frames=int(((mode>0)&~reliable).sum()),
            normal_violating_frames=np.flatnonzero(normal_bad).tolist(),fixed_xy_violating_frames=np.flatnonzero(xy_bad).tolist(),
            max_trusted_normal_m=float(np.nanmax(normal[reliable])) if reliable.any() and np.isfinite(normal[reliable]).any() else None,
            max_trusted_fixed_xy_m=float(np.nanmax(xy[reliable&(mode==1)])) if np.isfinite(xy[reliable&(mode==1)]).any() else None)
    return dict(policy=POLICY,normal_tolerance_m=normal_limit,fixed_xy_tolerance_m=xy_limit,
                violating_frames=np.flatnonzero(bad).tolist(),contacts=report,
                scope='inferred high-confidence contact geometry, not measured forces; sliding/rolling do not lock XY')
