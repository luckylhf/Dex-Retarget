"""Local error summaries for review; these do not silently change A/B policy."""
import numpy as np


def local_windows(values, fps, *, seconds=4., stride_seconds=.5):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or fps<=0:
        raise ValueError('Expected finite per-frame errors and positive FPS')
    width=min(len(values),max(1,round(seconds*fps)))
    starts=sorted(set(range(0,len(values)-width+1,max(1,round(stride_seconds*fps))))|{len(values)-width})
    return [dict(start_frame=a,end_frame_exclusive=a+width,start_s=a/fps,
                 end_s=(a+width-1)/fps,p95=float(np.percentile(values[a:a+width],95)),
                 max=float(np.max(values[a:a+width]))) for a in starts]


def compare_local_errors(old,new,fps,*,increase_deg=5.,absolute_deg=10.):
    old=np.asarray(old,dtype=float);new=np.asarray(new,dtype=float)
    if old.shape!=new.shape:raise ValueError('Comparison requires equal full-file timestamps')
    a=local_windows(old,fps);b=local_windows(new,fps)
    changes=[dict(start_s=x['start_s'],end_s=x['end_s'],old_p95_deg=x['p95'],
                  new_p95_deg=y['p95'],increase_deg=y['p95']-x['p95']) for x,y in zip(a,b)]
    material=[v for v in changes if v['increase_deg']>increase_deg and v['new_p95_deg']>absolute_deg]
    # Keep a compact set of non-overlapping examples rather than hundreds of flags.
    selected=[]
    for v in sorted(material,key=lambda x:x['increase_deg'],reverse=True):
        if all(v['end_s']<p['start_s'] or v['start_s']>p['end_s'] for p in selected):selected.append(v)
        if len(selected)==3:break
    return dict(policy='full_clip_local_4s_review_v1',material_window_count=len(material),
                review_windows=selected,worst_increase=max(changes,key=lambda x:x['increase_deg']))
