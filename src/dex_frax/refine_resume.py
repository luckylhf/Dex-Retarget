"""Identity-checked continuation from an exported refine checkpoint."""
import json
from pathlib import Path
import numpy as np


def locate(value):
    p=Path(value).resolve()
    if p.is_file():p=p.parent
    for folder in (p,p/'frax'):
        if (folder/'refine_state.json').exists():return folder
        if (folder/'identity.json').exists() and (folder/'run.json').exists():return folder
    raise ValueError('continue-from requires a verified fast result or refine checkpoint')


def verify(folder,identity,recheck=False):
    from .cli import sha
    old=json.loads((folder/'identity.json').read_text())
    if recheck:
        safety={'acceptance.py','audit_cache.py','support_quality.py','temporal_quality.py',
                'v8/audit.py','v8/anatomy.py','v8/robot.py','v8/self_collision.py'}
        names=set(old['implementation'])|set(identity['implementation'])
        for name in names:
            if name.endswith('.json') or name in safety:
                if old['implementation'].get(name)!=identity['implementation'].get(name):
                    raise ValueError('Recheck safety policy changed: '+name)
        run=json.loads((folder/'run.json').read_text())
        for name in ('motion.json','motion.diagnostics.npz','quality.json'):
            if run.get('files',{}).get(name)!=sha(folder/name):raise ValueError('Recheck artifact hash changed: '+name)
    for key in ('input_sha256','assets','implementation','frax_implementation','runtime'):
        if recheck and key=='implementation':continue
        if old[key]!=identity[key]:raise ValueError('Continuation identity mismatch: '+key)
    fast=old['conversion_contract'].get('profile')=='fast'
    def contract(value):
        result=dict(value['conversion_contract']);result.pop('repair_budget',None)
        if fast:result.pop('profile',None);result.pop('unified_repair',None)
        return result
    if identity['conversion_contract'].get('profile')!='refine':raise ValueError('Continuation target must be refine')
    if contract(old)!=contract(identity):raise ValueError('Continuation conversion contract changed')
    if fast:
        run=json.loads((folder/'run.json').read_text())
        if run.get('status')!='complete':raise ValueError('Fast source is not complete')
        for name in ('motion.pkl','motion.initial.npz','motion.json','motion.diagnostics.npz','quality.json'):
            if run.get('files',{}).get(name)!=sha(folder/name):raise ValueError('Fast artifact hash changed: '+name)
        return {'from_fast':True}
    state=json.loads((folder/'refine_state.json').read_text())
    if state['output_sha256']!=sha(folder/'motion.pkl'):raise ValueError('Checkpoint motion hash changed')
    if state['baseline_sha256']!=sha(folder/'frax_baseline/motion.pkl'):raise ValueError('Fixed Frax reference hash changed')
    if state.get('schema')!=3:raise ValueError('Continuation requires current dependency queue schema')
    if state['avoidance_reference_sha256']!=sha(folder/'avoidance_reference.npy'):raise ValueError('Avoidance reference changed')
    return state


class Initial:
    improved=False
    strict=True
    def __init__(self,folder):
        self.folder=folder
        self.meta=json.loads((folder/'motion.json').read_text());self.report=dict(self.meta['frax'])
        self.report['continued_without_ik']=True
    def __call__(self,robot,t,cfg,progress):
        from .initial_reuse import target_digest
        if target_digest(t)!=self.meta['prepared_targets_sha256']:raise ValueError('Continuation targets changed')
        baseline=self.folder/'frax_baseline' if (self.folder/'refine_state.json').exists() else self.folder
        with np.load(baseline/'motion.initial.npz',allow_pickle=False) as data:
            if not np.array_equal(data['joint_names'],robot.names):raise ValueError('Continuation joints changed')
            return data['x'].copy(),data['solver_records'].copy()
