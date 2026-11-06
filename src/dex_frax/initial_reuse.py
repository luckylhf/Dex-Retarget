"""Reuse only a verified, pre-refinement frax trajectory for hybrid."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def target_digest(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject:
                raise ValueError('Object arrays cannot be fingerprinted')
            h.update(b'array');h.update(str(v.dtype).encode());h.update(str(v.shape).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for key in sorted(v):visit(key);visit(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(b'list');h.update(str(len(v)).encode())
            for item in v:visit(item)
        else:
            if isinstance(v, np.generic):v = v.item()
            h.update(json.dumps(v, sort_keys=True, allow_nan=False).encode());h.update(b'\0')
    visit(value)
    return h.hexdigest()


class ReusedInitial:
    improved = True

    def __init__(self, directory, identity, fallback, strict=False):
        self.directory = Path(directory)
        self.identity = identity
        self.fallback = fallback
        self.report = {}
        self.strict=strict

    def __call__(self, robot, targets, config, progress=None):
        from .cli import sha
        try:
            root = self.directory
            previous = json.loads((root/'identity.json').read_text())
            run = json.loads((root/'run.json').read_text())
            if previous['mode'] != 'frax' or previous.get('legacy_frax') or run['status'] != 'complete':
                raise ValueError('Source is not a completed improved frax conversion')
            for key in ('converter','version','source','fps','iterations','initial_iterations',
                        'max_frames','input_sha256','conversion_contract','implementation','frax_implementation','assets','runtime'):
                if previous[key] != self.identity[key]:
                    raise ValueError('Initial identity differs: '+key)
            if previous.get('ik_block_frames',128)!=self.identity.get('ik_block_frames',128):
                raise ValueError('Initial execution block length differs')
            for name in ('motion.initial.npz','motion.json'):
                if sha(root/name) != run['files'][name]:
                    raise ValueError('Initial artifact hash differs: '+name)
            meta = json.loads((root/'motion.json').read_text())
            if meta['prepared_targets_sha256'] != target_digest(targets):
                raise ValueError('Prepared targets differ')
            if meta['configuration']['solver'] != config:
                raise ValueError('Solver configuration differs')
            with np.load(root/'motion.initial.npz', allow_pickle=False) as saved:
                x = saved['x'].copy();records = saved['solver_records'].copy()
                if (x.shape != (len(targets['hips']),32) or records.shape != (len(x),3)
                        or not np.isfinite(x).all() or not np.isfinite(records).all()
                        or not np.array_equal(saved['root_r'],targets['root_r'])
                        or saved['fps'].item() != targets['fps']
                        or saved['joint_names'].tolist() != list(robot.names)):
                    raise ValueError('Initial trajectory schema/targets differ')
            transitions = self.report.get('contact_transitions')
            self.report = copy.deepcopy(meta['frax'])
            for key in ('local_refinement','adaptive_refinement','runtime_cache'):
                self.report.pop(key,None)
            self.report.update(prepare_s=0.,compile_s=0.,solve_s=0.,total_s=0.,compile_queue_wait_s=0.,
                contact_transitions=transitions,
                initial_reuse=dict(used=True,source=str(root),fingerprint=previous['fingerprint']))
            self.initial = x.copy()
            from .resource_events import phase
            phase('initial_reused')
            return x,records.tolist()
        except (OSError,ValueError,KeyError,TypeError) as error:
            if self.strict:raise ValueError('Verified initial reuse failed; V8 is the independent fallback: '+str(error)) from error
            # Missing, stale or partial checkpoints never become solver input.
            transitions = self.report.get('contact_transitions')
            x,records = self.fallback(robot,targets,config,progress)
            self.report = self.fallback.report
            self.report['contact_transitions'] = transitions
            self.report['initial_reuse'] = dict(used=False,source=str(self.directory),reason=str(error))
            return x,records
