"""Measured collision ownership and strict local torso dependencies."""
import copy
import numpy as np


def descendant_bodies(robot,columns):
    roots=set(int(robot.m.jnt_bodyid[robot.jids[j-3]]) for j in columns)
    def belongs(body):
        while body:
            if body in roots:return True
            body=int(robot.m.body_parentid[body])
        return False
    return {b for b in range(1,robot.m.nbody) if belongs(b)}


def collision_groups(model):
    robot=model.r
    arms=descendant_bodies(robot,sum([g['arm'] for g in robot.sides.values()],[]))
    legs=descendant_bodies(robot,sum([g['leg'] for g in robot.sides.values()],[]))
    result={'collision':[],'leg_collision':[],'body_collision':[]}
    for k,(a,b) in enumerate(model.pairs):
        bodies={int(model.m.geom_bodyid[a]),int(model.m.geom_bodyid[b])}
        key='collision' if bodies&arms else 'leg_collision' if bodies&legs else 'body_collision'
        result[key].append(k)
    return result


def submodel(model,indices):
    """Solver sees only its movable collision class; adoption still checks all."""
    import mink
    from .ankle_collision import AnkleCollisionAvoidanceLimit
    result=copy.copy(model)
    pairs=model.pairs[np.asarray(indices,int)]
    result.limit=AnkleCollisionAvoidanceLimit(result.m,[([int(a)],[int(b)]) for a,b in pairs],
        minimum_distance_from_collisions=result.cfg['margin_m'],collision_detection_distance=result.cfg['detection_m'])
    result.pairs=np.asarray(result.limit.geom_id_pairs,int)
    result.names=[[result.m.geom(a).name,result.m.geom(b).name] for a,b in result.pairs]
    result.configuration=mink.Configuration(result.m)
    return result


def collision_masks(samples,groups,n):
    out={key:np.zeros(n,bool) for key in groups};owner={i:key for key,ids in groups.items() for i in ids}
    for sample,(_,_,pairs) in enumerate(samples):
        if not pairs:continue
        frame,k=divmod(sample,8)
        for key in {owner[p] for p in pairs}:
            out[key][frame]=True
            if k:out[key][min(n-1,frame+1)]=True
    return out


def dependencies(jobs,torso,arm_collision,body_collision=None,pending=None):
    """Prior unresolved stages block overlapping downstream windows.

    Exhausted tasks remain blockers; they cannot disappear merely because no
    solver route is left. Independent windows can still make progress.
    """
    from .refine_queue import PHASE
    pending=jobs if pending is None else pending
    ready=[];blocked=[]
    for job in jobs:
        a,b=job['start'],job['end'];stage=job['module'];reason=None
        if stage=='body_collision':reason='no_certified_body_collision_module'
        elif PHASE[stage]>1 and np.any(torso[a:b]>20.):reason='waiting_for_torso'
        elif PHASE[stage]>1 and body_collision is not None and np.any(body_collision[a:b]):reason='waiting_for_body_geometry'
        elif stage=='arms' and np.any(arm_collision[a:b]):reason='waiting_for_arm_clearance'
        if reason is None:
            blockers=[j for j in pending if PHASE[j['module']]<PHASE[stage] and j['start']<b and j['end']>a]
            if blockers:reason='waiting_for_'+min(blockers,key=lambda j:PHASE[j['module']])['module']
        if reason:blocked.append(dict(job,reason=reason))
        else:ready.append(job)
    return ready,blocked


def alignment_reference(reference,aligned,unfixed,a,b):
    """Only an unresolved waist may establish a new pre-avoidance reference.

    Correct neighboring frames keep their original reference, even when a
    larger repair window overlaps them. Never rebase an already-correct frame
    on the outcome of previous shoulder avoidance.
    """
    out=reference.copy();indices=np.flatnonzero(np.asarray(unfixed)[a:b])+a
    out[indices]=aligned[indices]
    return out


def movable_submodel(model,columns):
    """All pairs affected by these joints, regardless of repair ownership.

    A leg repair owns leg/body penetration, but it can also push a thigh into a
    fixed hand. Such arm/leg pairs must constrain its candidate solver too.
    """
    bodies=descendant_bodies(model.r,columns)
    indices=[k for k,(a,b) in enumerate(model.pairs)
             if int(model.m.geom_bodyid[a]) in bodies or int(model.m.geom_bodyid[b]) in bodies]
    return submodel(model,indices)


SAFETY_ORDER={'leg_collision':0,'support':1,'dense_floor':2,'jitter':3,'body':4,'collision':5}


def ready_jobs(queue,pending,torso,arm_collision,body_collision,attempts):
    """Shared safety fits protect each other physically, without queue deadlocks.

    Identical selection is used by normal refinement and safety-scoped runs.
    Existing dependencies continue to protect pose-only work after safety.
    """
    actionable=queue.actionable(pending)
    safety=[job for job in actionable if job['module'] in SAFETY_ORDER]
    if safety:
        visits={name:sum(entry['module']==name for entry in attempts) for name in SAFETY_ORDER}
        chosen=min({job['module'] for job in safety},key=lambda name:(visits[name],SAFETY_ORDER[name]))
        return [job for job in safety if job['module']==chosen],[]
    return dependencies(actionable,torso,arm_collision,body_collision,pending=pending)
