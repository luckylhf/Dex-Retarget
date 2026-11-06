"""Validated SMPL-X parameters and CPU geometry. No robot solver imports here.

PT is weights-only; PKL permits only numeric NumPy reconstruction. A file
extension is a container choice, not a skeleton or coordinate convention.
"""
import argparse
import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np


class InputError(ValueError):
    pass


class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return getattr(np.core.multiarray, name)
        if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
            return np.core.numeric._frombuffer
        raise InputError(f'Unsupported PKL object {module}.{name}; export a plain numeric NPZ instead')

    def persistent_load(self, pid):
        raise InputError('External PKL references are not supported; export numeric NPZ')


def _plain(value):
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise InputError('Input dictionaries require string keys')
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufUS':
            raise InputError(f'Unsupported array dtype {value.dtype}; object arrays are forbidden')
        return value
    if isinstance(value, (str, bytes, int, float, bool, np.generic)) or value is None:
        return value
    raise InputError(f'Unsupported input value {type(value).__name__}')


def read_container(path):
    path = Path(path)
    try:
        if path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as data:
                result = {k: data[k] for k in data.files}
        elif path.suffix.lower() == '.pkl':
            result = NumericUnpickler(io.BytesIO(path.read_bytes())).load()
        elif path.suffix.lower() == '.pt':
            try:
                import torch
            except ImportError as exc:
                raise InputError('Reading PT requires torch; use --smplx-python or install the smplx extra') from exc
            result = torch.load(path, map_location='cpu', weights_only=True)
        else:
            raise InputError('SMPL-X input must be .npz, .pt or .pkl')
        result = _plain(result)
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f'Cannot safely read {path.name}: {exc}. Export numeric NPZ; unsafe loading is not attempted') from exc
    if not isinstance(result, dict):
        raise InputError('Expected a dictionary of SMPL-X fields')
    if any(k in result for k in ('dof_pos', 'joint_pos', 'body_pos_w')):
        raise InputError('This is robot motion, not SMPL-X human parameters')
    return result


def pick(data, names, required=False):
    present = [k for k in names if k in data]
    if not present:
        if required:
            raise InputError('Missing field: ' + ' / '.join(names))
        return None
    result = data[present[0]]
    if any(not np.array_equal(np.asarray(result), np.asarray(data[k])) for k in present[1:]):
        raise InputError('Conflicting aliases: ' + ', '.join(present))
    return result


def scalar(value, field):
    a = np.asarray(value)
    if a.size != 1:
        raise InputError(f'{field} must be a scalar')
    v = a.reshape(()).item()
    return v.decode() if isinstance(v, bytes) else v


def numeric(value, field):
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise InputError(f'{field} must contain finite numeric values')
    return a.astype(np.float64)


def normalize(data, *, input_fps=None, up_axis=None, units=None):
    gvhmr = 'smpl_params_global' in data
    if 'smpl_params_incam' in data and not gvhmr:
        raise InputError('Camera-space SMPL-X cannot be used as world motion; provide smpl_params_global')
    top = data
    data = data['smpl_params_global'] if gvhmr else data
    if not isinstance(data, dict):
        raise InputError('smpl_params_global must be a dictionary')
    report = {'schema': 'smplx_parameters_v1', 'producer': 'gvhmr' if gvhmr else 'generic', 'warnings': []}
    for key in ('model_type', 'surface_model_type'):
        if key in data and str(scalar(data[key], key)).lower() != 'smplx':
            raise InputError(f'{key} must be smplx; SMPL/SMPL-H require a separate conversion')
    fps_value = pick(top, ('fps', 'mocap_framerate', 'mocap_frame_rate', 'mocap_fps'))
    if fps_value is not None:
        fps_value = float(scalar(fps_value, 'fps'))
        if input_fps is not None and not np.isclose(fps_value, input_fps):
            raise InputError('--input-fps conflicts with file metadata')
    else:
        fps_value = input_fps
    if fps_value is None or not np.isfinite(fps_value) or fps_value <= 0:
        raise InputError('Missing/invalid fps; supply --input-fps (normalized GVHMR video is 30 fps)')
    declared_axis = str(scalar(top['up_axis'], 'up_axis')).lower() if 'up_axis' in top else ('y' if gvhmr else None)
    if up_axis and declared_axis and up_axis != declared_axis:
        raise InputError('--up-axis conflicts with source convention')
    axis = up_axis or declared_axis
    if axis not in ('y', 'z'):
        raise InputError('Missing up_axis; supply --up-axis y or z. Orientation is never guessed from a pose')
    declared_units = str(scalar(top['units'], 'units')).lower() if 'units' in top else ('m' if gvhmr else None)
    if units and declared_units and units != declared_units:
        raise InputError('--units conflicts with file metadata')
    if (units or declared_units) != 'm':
        raise InputError('Translations must be metres; supply units="m" or --units m after checking the source')
    if 'pose_representation' in data and scalar(data['pose_representation'], 'pose_representation') != 'axis_angle':
        raise InputError('Only local axis-angle poses in radians are supported (no PCA/quaternion/matrix inputs)')
    combined = pick(data, ('poses', 'pose', 'smpl_poses', 'thetas'))
    fields = {}
    if combined is not None:
        pose = numeric(combined, 'poses')
        if pose.ndim != 2 or pose.shape[1] not in (66, 165):
            raise InputError('poses must be (T,66) root+body or (T,165) SMPL-X root+body+jaw+eyes+hands; 72/156 are not SMPL-X')
        fields.update(global_orient=pose[:, :3], body_pose=pose[:, 3:66])
        if pose.shape[1] == 165:
            fields.update(jaw_pose=pose[:,66:69], leye_pose=pose[:,69:72], reye_pose=pose[:,72:75],
                          left_hand_pose=pose[:,75:120], right_hand_pose=pose[:,120:165])
    aliases = {'global_orient': ('global_orient','root_orient'), 'body_pose': ('body_pose','pose_body'),
               'jaw_pose': ('jaw_pose','pose_jaw'), 'leye_pose': ('leye_pose',), 'reye_pose': ('reye_pose',),
               'left_hand_pose': ('left_hand_pose',), 'right_hand_pose': ('right_hand_pose',)}
    for key, names in aliases.items():
        value = pick(data, names)
        if value is not None:
            value = numeric(value, key)
            if key in fields and not np.array_equal(fields[key], value):
                raise InputError(f'Conflicting combined and split pose field: {key}')
            fields[key] = value
    for key, targets, widths in [('pose_hand',('left_hand_pose','right_hand_pose'),(45,45)),
                                  ('pose_eye',('leye_pose','reye_pose'),(3,3))]:
        if key in data:
            value = numeric(data[key], key)
            if value.ndim != 2 or value.shape[1] != sum(widths):
                raise InputError(f'{key} must be (T,{sum(widths)}); PCA hands are unsupported')
            for name, part in zip(targets, np.split(value, [widths[0]], axis=1)):
                if name in fields and not np.array_equal(fields[name], part):
                    raise InputError(f'Conflicting hand/eye aliases: {key}, {name}')
                fields[name] = part
    if 'global_orient' not in fields or 'body_pose' not in fields:
        raise InputError('Need global_orient/root_orient (T,3) and body_pose/pose_body (T,63)')
    if fields['global_orient'].ndim != 2:
        raise InputError('global_orient/root_orient must be (T,3)')
    n = len(fields['global_orient'])
    if n < 3:
        raise InputError('Motion needs at least 3 frames')
    for key, value in fields.items():
        width = 63 if key == 'body_pose' else (45 if 'hand' in key else 3)
        if value.shape != (n, width):
            raise InputError(f'{key} must be ({n},{width}), got {value.shape}')
    fields['transl'] = numeric(pick(data, ('transl','trans','translations'), required=True), 'transl')
    if fields['transl'].shape != (n,3):
        raise InputError(f'transl must be ({n},3)')
    beta = pick(data, ('betas','shape'))
    if beta is None:
        beta = np.zeros(10)
        report['warnings'].append('missing_betas_using_zero_shape')
    beta = numeric(beta,'betas')
    if beta.ndim == 1:
        beta = beta[None]
    if beta.ndim != 2 or beta.shape[0] not in (1,n) or not 1 <= beta.shape[1] <= 16:
        raise InputError('betas must be (B,), (1,B) or (T,B), 1<=B<=16')
    if len(beta) == n and not np.allclose(beta,beta[:1],atol=1e-5,rtol=0):
        report['warnings'].append('time_varying_betas_preserved_sole_regions_use_median_shape')
    fields['betas'] = np.broadcast_to(beta,(n,beta.shape[1])).copy()
    fields['expression'] = numeric(data.get('expression',np.zeros((n,10))), 'expression')
    if fields['expression'].shape != (n,10):
        raise InputError('expression must be (T,10)')
    gender = str(scalar(data.get('gender',top.get('gender','neutral')), 'gender')).lower()
    if gender not in ('neutral','male','female'):
        raise InputError('gender must be neutral, male or female')
    missing_hands = [k for k in ('left_hand_pose','right_hand_pose') if k not in fields]
    if missing_hands:
        report['warnings'].append('unobserved_hands_use_model_relaxed_mean')
    conf = None
    if 'static_confidence' in top:
        conf = numeric(top['static_confidence'],'static_confidence')
        if conf.shape != (n,4) or np.any((conf<0)|(conf>1)):
            raise InputError('static_confidence must be (T,4) probabilities [left ankle,left foot,right ankle,right foot]')
    elif gvhmr and 'static_conf_logits' in top.get('net_outputs',{}):
        logits = numeric(top['net_outputs']['static_conf_logits'],'static_conf_logits')
        if logits.shape == (1,n,6): logits = logits[0]
        if logits.shape != (n,6): raise InputError('GVHMR static_conf_logits must be (1,T,6) or (T,6)')
        conf = 1/(1+np.exp(-np.clip(logits[:,:4],-60,60)))
    report.update(frames=n,fps=float(fps_value),up_axis=axis,units='m',gender=gender,
                  pose_representation='axis_angle_radians',hands_observed=not missing_hands,
                  static_confidence_available=conf is not None,
                  static_confidence_semantics='joint stationarity, not measured ground contact')
    return fields, conf, report


def calibrate_height(arrays, confidence, fps):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    if confidence is None:
        raise InputError('Contact grounding requires static_confidence or GVHMR static_conf_logits; use --grounding off otherwise')
    feet = np.stack([arrays['Left_all'],arrays['Right_all']],axis=1)
    prob = np.stack([confidence[:,:2].max(1),confidence[:,2:].max(1)],axis=1)
    valid = (prob>.8)&(feet<=feet.min(1,keepdims=True)+.03)
    weights = valid*prob**4
    w = weights.sum(1)
    if np.mean(w>0)<=.1 or np.count_nonzero(w)<3:
        raise InputError('Insufficient static support evidence for height correction; no ground is invented')
    n = len(w)
    d2 = sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)).tocsr()
    lam = (.20*fps)**4
    drift = spsolve((sparse.diags(w+1e-8)+lam*(d2.T@d2)).tocsc(),(weights*feet).sum(1))
    arrays['positions'][:,:,2] -= drift[:,None]
    for side in ('Left','Right'):
        for key in ('heel','toe'): arrays[side+'_'+key][:,2] -= drift
        arrays[side+'_all'] -= drift
    arrays.update(vertical_drift=drift,static_support_evidence=valid,drift_observed=w>0)
    return dict(method='stationarity_weighted_common_vertical_drift_v1',support_observed_fraction=float(np.mean(w>0)),
                correction_minmax_m=[float(drift.min()),float(drift.max())],smoothing_seconds=.2,
                airborne_handling='no per-frame ground pin; drift interpolated through unsupported intervals',
                measured_contact=False,automatic_jump_preservation_verified=False)


def prepare(path, output, body_models, *, input_fps=None, up_axis=None, units=None,
            grounding='off', max_frames=None):
    from scipy.spatial.transform import Rotation, Slerp
    try:
        import torch
        import smplx
        from smplx.joint_names import JOINT_NAMES
    except ImportError as exc:
        raise InputError('SMPL-X geometry requires torch + smplx; configure --smplx-python or install .[smplx]') from exc
    fields, conf, report = normalize(read_container(path),input_fps=input_fps,up_axis=up_axis,units=units)
    body_models = Path(body_models).resolve()
    model_path = body_models/'smplx'/('SMPLX_'+report['gender'].upper()+'.npz')
    if not model_path.is_file():
        raise InputError(f'Missing {model_path}; obtain licensed SMPL-X NPZ models from https://smpl-x.is.tue.mpg.de/')
    fps = 50.
    native = np.arange(report['frames'])/report['fps']
    times = np.arange(int(np.floor(native[-1]*fps+1e-8))+1)/fps
    if max_frames is not None: times = times[:max_frames]
    if len(times)<3: raise InputError('Resampled motion needs at least 3 frames')
    sampled = {}
    for key,value in fields.items():
        if key in ('transl','betas','expression'):
            sampled[key] = np.stack([np.interp(times,native,value[:,j]) for j in range(value.shape[1])],axis=1)
        else:
            value = value.reshape(len(value),-1,3)
            sampled[key] = np.stack([Slerp(native,Rotation.from_rotvec(value[:,j]))(times).as_rotvec()
                                     for j in range(value.shape[1])],axis=1).reshape(len(times),-1)
    torch.set_num_threads(2)
    model = smplx.create(str(body_models),'smplx',gender=report['gender'],use_pca=False,num_betas=16,ext='npz',flat_hand_mean=False).eval()
    parents = model.parents.detach().cpu().numpy()
    names = JOINT_NAMES[:len(parents)]
    transform = np.array([[1.,0,0],[0,0,-1],[0,1,0]]) if report['up_axis']=='y' else np.eye(3)
    def call(values):
        n = len(values['transl'])
        kwargs = {}
        for key,width in [('global_orient',3),('body_pose',63),('jaw_pose',3),('leye_pose',3),('reye_pose',3),
                          ('left_hand_pose',45),('right_hand_pose',45),('expression',10),('transl',3)]:
            a = values.get(key,np.zeros((n,width)))
            # Explicit hands are absolute local axis angles; missing hands use the relaxed model mean.
            if key in values and key in ('left_hand_pose','right_hand_pose'):
                mean = model.left_hand_mean if key=='left_hand_pose' else model.right_hand_mean
                a = a-mean.detach().cpu().numpy()
            kwargs[key] = torch.tensor(a,dtype=torch.float32)
        beta = values['betas']; beta = np.pad(beta,((0,0),(0,16-beta.shape[1])))
        return model(**kwargs,betas=torch.tensor(beta,dtype=torch.float32),return_full_pose=True)
    with torch.no_grad():
        neutral = call({'transl':np.zeros((1,3)),'betas':np.median(fields['betas'],axis=0,keepdims=True)})
    vn = neutral.vertices[0].numpy(); jn = neutral.joints[0].numpy()
    weights = model.lbs_weights.numpy(); regions={}
    for side,ankle,toe in [('Left',7,10),('Right',8,11)]:
        ids = np.flatnonzero(weights[:,[ankle,toe]].sum(1)>.5)
        sole = ids[vn[ids,1]<vn[ids,1].min()+.012]
        regions[side] = dict(all=ids,heel=sole[vn[sole,2]<=np.percentile(vn[sole,2],25)],toe=sole[vn[sole,2]>=np.percentile(vn[sole,2],75)])
    chunks={}; joints=[]; rotations=[]
    for start in range(0,len(times),32):
        with torch.no_grad(): result = call({k:v[start:start+32] for k,v in sampled.items()})
        p = result.joints.numpy()[:,:len(parents)]@transform.T
        v = result.vertices.numpy()@transform.T
        local = Rotation.from_rotvec(result.full_pose.numpy().reshape(-1,3)).as_matrix().reshape(len(p),-1,3,3)
        r = np.empty_like(local)
        for j,parent in enumerate(parents): r[:,j] = transform@local[:,j] if parent<0 else r[:,parent]@local[:,j]
        joints.append(p); rotations.append(r)
        for side,regions_side in regions.items():
            for key,ids in regions_side.items():
                val = v[:,ids,2].min(1) if key=='all' else v[:,ids].mean(1)
                chunks.setdefault(side+'_'+key,[]).append(val)
    # Neutral anatomy is always represented in canonical Z-up, independently of animated input coordinates.
    canonical = np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    arrays = dict(positions=np.concatenate(joints),rotations=np.concatenate(rotations),
                  neutral_positions=jn[:len(parents)]@canonical.T,names=np.array(names),times=times,fps=fps,
                  **{k:np.concatenate(v) for k,v in chunks.items()})
    if grounding=='contact':
        interp_conf = None if conf is None else np.stack([np.interp(times,native,conf[:,j]) for j in range(4)],axis=1)
        report['grounding'] = calibrate_height(arrays,interp_conf,fps)
        report['warnings'].append('contact_grounding_is_inferred_review_jumps_and_long_unobserved_intervals')
    else:
        report['grounding'] = {'method':'off'}
    import importlib.metadata
    report.update(input=str(Path(path).resolve()),input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  body_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),body_model=str(model_path),
                  output_frames=len(times),output_fps=fps,
                  runtime={k:importlib.metadata.version(k) for k in ('torch','smplx','numpy','scipy')},
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    arrays['metadata_json'] = np.array(json.dumps(report,sort_keys=True))
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream: np.savez_compressed(stream,**arrays)
    output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--body-models',type=Path)
    parser.add_argument('--input-fps',type=float)
    parser.add_argument('--up-axis',choices=['y','z'])
    parser.add_argument('--units',choices=['m'])
    parser.add_argument('--grounding',choices=['off','contact'],default='off')
    parser.add_argument('--max-frames',type=int)
    args = parser.parse_args()
    try:
        if args.output:
            if not args.body_models: raise InputError('--body-models is required for geometry preparation')
            report=prepare(args.input,args.output,args.body_models,input_fps=args.input_fps,up_axis=args.up_axis,
                           units=args.units,grounding=args.grounding,max_frames=args.max_frames)
        else:
            _,conf,report=normalize(read_container(args.input),input_fps=args.input_fps,up_axis=args.up_axis,units=args.units)
            if args.grounding=='contact' and conf is None: raise InputError('Contact grounding requires static confidence')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except (InputError,OSError) as exc:
        parser.exit(2,f'SMPL-X input error: {exc}\n')


if __name__=='__main__':
    main()
