import numpy as np
from dex_frax.support_quality import preserve_contact,evaluate_contacts
from dex_frax.refine_pipeline import improved


def test_internal_contact_locks_match_final_trusted_frames():
    contact=dict(body_name='knee',mode=np.ones(3,int),confidence=np.array([.25,.8,1.]),source_geometry_confidence=1.)
    meta=dict(leg_scale=1.,configuration=dict(contact={}),body_contact_models=[contact])
    dg=dict(body_knee_mode=contact['mode'],body_knee_confidence=contact['confidence'])
    errors=dict(knee=dict(normal_cm=[10.,10.,10.],fixed_xy_cm=[0.,0.,0.]))
    final=evaluate_contacts(meta,dg,errors,3,50.)
    assert final['violating_frames']==[1,2]
    assert [i for i in range(3) if preserve_contact(contact,i)]==[1,2]
    contact['source_geometry_confidence']=.25
    assert not any(preserve_contact(contact,i) for i in range(3))
    assert preserve_contact(dict(mode=np.ones(3,int)),0)


def test_fixing_one_jitter_joint_counts_even_when_another_shares_the_frames():
    old=dict(temporal_quality=dict(violating_frames=[10,11],joints={'wrist_pitch_r_joint':dict(violating_frames=[10,11]),'wrist_roll_r_joint':dict(violating_frames=[10,11])}),flags=['conversion_added_jitter'])
    new=dict(temporal_quality=dict(violating_frames=[10,11],joints={'wrist_pitch_r_joint':dict(violating_frames=[]),'wrist_roll_r_joint':dict(violating_frames=[10,11])}),flags=['conversion_added_jitter'])
    assert improved('jitter','wrist_pitch_r_joint',0,20,old,new,{}, {})
