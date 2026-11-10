"""Collision-type preferences, never different per-axis angular limits."""
import numpy as np
from scipy.spatial.transform import Rotation


def preferences(kinds,contact):
    """Positive posture costs for J1..J6; every joint remains available."""
    prior=np.array([.12,.12,.14,.18,.14,.14])
    if 'cross_leg' in kinds:prior=np.minimum(prior,[.08,.04,.04,.10,.14,.14])
    if 'own_foot_shin' in kinds:prior=np.minimum(prior,[.14,.14,.16,.18,.04,.04])
    if 'leg_body' in kinds:prior=np.minimum(prior,[.06,.06,.10,.10,.12,.12])
    # A stance leg may cooperate, but a free swing leg should move first.
    return prior*(2. if contact else 1.)


def plan(model,t,rows,a,b,ids):
    from .refine_routing import collision_groups,descendant_bodies
    r=model.r;n=b-a
    owners={s:descendant_bodies(r,g['leg']) for s,g in r.sides.items()}
    kinds={s:[set() for _ in range(n)] for s in r.sides};owned=collision_groups(model)['leg_collision']
    counts={}
    rotations=t['root_r'][a:b]
    delta=Rotation.from_matrix(rotations[:-1].transpose(0,2,1)@rotations[1:]).as_rotvec() if n>1 else []
    for j in range(n):
        for part in range(8 if j+1<n else 1):
            alpha=part/8
            row=rows[a+j] if not part else rows[a+j]*(1-alpha)+rows[a+j+1]*alpha
            rotation=rotations[j] if not part else rotations[j]@Rotation.from_rotvec(alpha*delta[j]).as_matrix()
            model.update(row,rotation);distance=model.distances()
            for k in owned:
                if distance[k]>=-model.cfg['penetration_tolerance_m']:continue
                bodies={int(model.m.geom_bodyid[g]) for g in model.pairs[k]}
                sides=[s for s in r.sides if owners[s]&bodies]
                if len(sides)==2:kind='cross_leg'
                elif len(sides)==1 and bodies=={r.sides[sides[0]]['foot'],r.sides[sides[0]]['knee']}:kind='own_foot_shin'
                else:kind='leg_body'
                counts[kind]=counts.get(kind,0)+1
                for side in sides:
                    for h in range(max(0,j-3),min(n,j+5)):kinds[side][h].add(kind)
    prior=np.full((n,len(ids)),1.);index={int(v):k for k,v in enumerate(ids)}
    for side,g in r.sides.items():
        columns=[index[k] for k in g['leg']]
        for j in range(n):prior[j,columns]=preferences(kinds[side][j],t['sides'][side]['contact'][a+j]>0)
    return prior,dict(policy='fixed_reference_5_10_joint_leg_preferences_v1',collision_type_samples=counts,
        own_foot_shin_primary=['J5','J6'],cross_leg_primary=['J2','J3'],all_joints_available=True,
        pelvis_compensation='height_only',support_mode='preserve_original_contact_pivot_and_anchor',swing_leg_preferred=True)
