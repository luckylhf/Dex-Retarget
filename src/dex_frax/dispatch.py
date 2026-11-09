"""Deterministic duration routing; the BVH header is the only probe."""
from decimal import Decimal

STAGES=('frax','v8')


def deduplicate(cases):
    unique=[];seen={};aliases=[]
    for case in cases:
        key=case['input_sha256']
        if key not in seen:seen[key]=case;unique.append(case);continue
        canonical=seen[key]
        if canonical['source']!=case['source']:raise ValueError('Identical BVH has conflicting source profiles')
        aliases.append(dict(id=case['id'],canonical=canonical['id'],input=case['input'],input_sha256=key))
    return unique,aliases


def duration_seconds(path):
    frames=dt=None
    with open(path,encoding='utf-8') as stream:
        for line in stream:
            text=line.strip()
            if text.startswith('Frames:'):frames=int(text.split(':',1)[1])
            if text.startswith('Frame Time:'):
                dt=Decimal(text.split(':',1)[1].strip());break
    if frames is None or frames<2 or dt is None or not dt.is_finite() or dt<=0:
        raise ValueError('Invalid BVH duration header')
    return Decimal(frames)*dt


def route(mode,duration,single=False):
    if mode not in ('auto', *STAGES):raise ValueError('Unknown conversion mode: '+str(mode))
    return STAGES if mode=='auto' else (mode,)


def priority(entry,durations):
    index,mode=entry
    # Short Frax first; the scheduler drains all Frax before longest-first V8.
    return (STAGES.index(mode), -durations[index] if mode=='v8' else durations[index])
