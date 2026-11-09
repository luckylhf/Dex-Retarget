import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from dex_frax.smplx_input import InputError, normalize, read_container, calibrate_height


def motion(n=20):
    return dict(root_orient=np.zeros((n,3)),pose_body=np.zeros((n,63)),trans=np.zeros((n,3)),
                fps=30.,up_axis='y',units='m',betas=np.zeros(10),gender='neutral')


@pytest.mark.parametrize('suffix',['.npz','.pkl'])
def test_numeric_formats_and_split_fields(tmp_path,suffix):
    data=motion();path=tmp_path/('human'+suffix)
    if suffix=='.npz':np.savez(path,**data)
    else:path.write_bytes(pickle.dumps(data,protocol=5))
    fields,conf,report=normalize(read_container(path))
    assert fields['body_pose'].shape==(20,63) and report['fps']==30 and conf is None
    assert 'unobserved_hands_use_model_relaxed_mean' in report['warnings']


def test_gvhmr_needs_explicit_fps_and_preserves_confidence():
    m=motion();data=dict(smpl_params_global=dict(global_orient=m['root_orient'],body_pose=m['pose_body'],transl=m['trans'],betas=m['betas']),
                        net_outputs=dict(static_conf_logits=np.zeros((1,20,6))))
    with pytest.raises(InputError,match='fps'):normalize(data)
    _,conf,report=normalize(data,input_fps=30)
    assert conf.shape==(20,4) and np.all(conf==.5) and report['up_axis']=='y'


@pytest.mark.parametrize('mutator,match',[
    (lambda d:d.update(poses=np.zeros((20,72))),'72/156'),
    (lambda d:d.update(model_type='smplh'),'SMPL-H'),
    (lambda d:d.update(pose_body=np.zeros((19,63))),'shape|must be'),
    (lambda d:d.update(trans=np.full((20,3),np.nan)),'finite'),
    (lambda d:d.update(units='cm'),'metres'),
    (lambda d:d.update(fps=0),'fps'),
    (lambda d:d.pop('up_axis'),'up_axis'),
    (lambda d:d.update(pose_hand=np.zeros((20,12))),'PCA'),
    (lambda d:d.update(global_orient=np.ones((20,3))),'Conflicting'),
    (lambda d:d.update(static_confidence=np.full((20,4),2.)),'probabilities'),
])
def test_bad_inputs_fail_before_geometry(mutator,match):
    data=motion();mutator(data)
    with pytest.raises(InputError,match=match):normalize(data)


def test_robot_motion_is_not_human(tmp_path):
    path=tmp_path/'robot.npz';np.savez(path,dof_pos=np.zeros((10,29)))
    with pytest.raises(InputError,match='robot motion'):read_container(path)


def test_pickle_does_not_execute_globals(tmp_path):
    # No malicious payload execution is attempted: the disallowed class is rejected.
    path=tmp_path/'custom.pkl';path.write_bytes(pickle.dumps(Path('/tmp/example')))
    with pytest.raises(InputError,match='Unsupported PKL object'):read_container(path)


def test_object_npz_rejected(tmp_path):
    path=tmp_path/'object.npz';np.savez(path,x=np.array([{}],dtype=object))
    with pytest.raises(InputError,match='safely read'):read_container(path)


def test_165_layout_retains_hands_and_rejects_conflicts():
    data=motion();data.pop('root_orient');data.pop('pose_body')
    data['poses']=np.arange(20*165).reshape(20,165)*.0001
    fields,_,report=normalize(data)
    assert np.array_equal(fields['left_hand_pose'],data['poses'][:,75:120])
    assert report['hands_observed']
    data['pose_body']=np.zeros((20,63))
    with pytest.raises(InputError,match='Conflicting'):normalize(data)


def test_height_correction_does_not_clamp_airborne_frames():
    n=100;z=np.full(n,.2);z[40:60]+=.15
    arrays=dict(positions=np.zeros((n,3,3)))
    arrays['positions'][:,:,2]=z[:,None]
    for side in ('Left','Right'):
        arrays[side+'_all']=z.copy()
        for key in ('heel','toe'):arrays[side+'_'+key]=np.column_stack([np.zeros(n),np.zeros(n),z])
    confidence=np.ones((n,4));confidence[40:60]=0
    report=calibrate_height(arrays,confidence,50.)
    assert np.min(arrays['Left_all'][42:58])>.14
    assert np.max(np.abs(arrays['Left_all'][:35]))<.001
    assert report['measured_contact'] is False


def test_no_confidence_does_not_guess_ground():
    with pytest.raises(InputError,match='requires'):calibrate_height({},None,50.)


def test_auto_smplx_selection_does_not_promote_review(tmp_path,monkeypatch):
    from dex_frax import smplx_cli
    path=tmp_path/'human.npz';np.savez(path,**motion())
    monkeypatch.setattr(smplx_cli,'prepare',lambda *a,**kw:None)
    class Worker:
        calls=[]
        def close(self):pass
        def run(self,job,timeout):
            assert job['arguments'].get('reuse_initial_from') is None
            assert not job['arguments'].get('require_initial_reuse',False)
            self.calls.append(job['arguments']['mode'])
            return dict(status='complete',quality='review',flags=['source_pose_mismatch'],result={})
    args=argparse.Namespace(input=path,output=tmp_path/'out',budget=10.)
    worker=Worker();record=smplx_cli.single_auto(args,lambda:worker)
    assert worker.calls==['frax','v8'] and record['selected_mode'] is None
    assert record['preview_output']=='v8' and record['training_npz'] is None
    assert json.loads((args.output/'accepted.json').read_text())['motions']==[]
