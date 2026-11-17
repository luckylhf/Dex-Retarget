"""Reload resource admission budgets without changing a motion's solver identity."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from . import resources


class ResourcePolicy:
    def __init__(self, args, out, ceiling):
        self.path = Path(getattr(args, 'resource_policy', None) or out/'resource_policy.json').resolve()
        self.events = out/'resource_policy_events.jsonl'
        self.ceiling = ceiling
        resident = getattr(args, 'worker_resident_gib', args.worker_peak_gib)
        per_frame = int(getattr(args, 'frame_memory_kib', 32)*1024)
        self.value = dict(schema=1, workers=ceiling,memory_learning=not getattr(args,'no_memory_learning',False),
            completion_worker_gib=getattr(args,'completion_worker_gib',None),phase_bases_gib={},
            compile_extra_gib=max(0., args.worker_peak_gib-resident),
            stages=dict(frax=dict(base_gib=resident, bytes_per_output_frame=per_frame),
                v8=dict(base_gib=getattr(args, 'v8_worker_peak_gib', args.worker_peak_gib), bytes_per_output_frame=per_frame)))
        self.revision = None
        self.signature = None
        self.checked = float('-inf')
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix+'.tmp')
            tmp.write_text(json.dumps(self.value, indent=2)+'\n')
            tmp.replace(self.path)
        self.reload(force=True, initial=True)

    def validate(self, value):
        required={'schema', 'workers', 'compile_extra_gib', 'stages'}
        if not required.issubset(value) or set(value)-required-{'memory_learning','completion_worker_gib','phase_bases_gib'} or value['schema'] != 1:
            raise ValueError('Invalid resource policy schema')
        if type(value.get('memory_learning',True)) is not bool:raise ValueError('Invalid memory learning flag')
        if type(value['workers']) is not int or not 1 <= value['workers'] <= self.ceiling:
            raise ValueError('Policy workers exceed the startup ceiling')
        def number(v, lower, upper):
            return type(v) in (int, float) and math.isfinite(v) and lower <= v <= upper
        if value.get('completion_worker_gib') is not None and not number(value['completion_worker_gib'],.25,4096.):
            raise ValueError('Invalid completion worker budget')
        phases=value.get('phase_bases_gib',{})
        from .resource_events import STAGES
        if not isinstance(phases,dict) or set(phases)-set().union(*STAGES.values()) or not all(number(v,.25,4096.) for v in phases.values()):
            raise ValueError('Invalid phase memory budgets')
        if not number(value['compile_extra_gib'], 0., 16.):
            raise ValueError('Invalid shared compiler budget')
        if set(value['stages']) != {'frax', 'v8'}:
            raise ValueError('Both Frax and V8 stage budgets are required')
        for stage in value['stages'].values():
            if set(stage) != {'base_gib', 'bytes_per_output_frame'}:
                raise ValueError('Invalid stage fields')
            if not number(stage['base_gib'], .25, 32.):
                raise ValueError('Invalid stage resident budget')
            if type(stage['bytes_per_output_frame']) is not int or not 0 <= stage['bytes_per_output_frame'] <= 1048576:
                raise ValueError('Invalid per-frame budget')

    def reload(self, force=False, initial=False):
        now = time.monotonic()
        if not force and now-self.checked < 2.: return False
        self.checked = now
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature == self.signature: return False
            self.signature = signature
            if stat.st_size > 65536: raise ValueError('Resource policy is too large')
            value = json.loads(self.path.read_text())
            self.validate(value)
            revision = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            if revision == self.revision: return False
            self.value = value
            self.revision = revision
            event = dict(at=time.time(), status='applied', revision=revision, policy=value)
        except (OSError, ValueError, TypeError, KeyError) as error:
            if initial: raise ValueError('Initial resource policy is invalid: '+str(error)) from error
            event = dict(at=time.time(), status='rejected', reason=str(error), retained_revision=self.revision)
        with self.events.open('a') as stream: stream.write(json.dumps(event, sort_keys=True)+'\n')
        return event['status'] == 'applied'

    def peak(self, mode, frames):
        stage = self.value['stages'][mode]
        base=max(stage['base_gib'],max(self.value.get('phase_bases_gib',{}).values(),default=0))
        return resources.peak_bytes(frames, base, stage['bytes_per_output_frame'])

    def stage_peak(self,mode,frames,phase):
        stage=self.value['stages'][mode]
        base=self.value.get('phase_bases_gib',{}).get(phase,stage['base_gib'])
        return resources.peak_bytes(frames,base,stage['bytes_per_output_frame'])

    def completion_peak(self,mode,frames):
        stage=self.value['stages'][mode]
        base=self.value.get('completion_worker_gib')
        if base is None:base=stage['base_gib']+(self.value['compile_extra_gib'] if mode=='frax' else 0)
        base=max(base,max(self.value.get('phase_bases_gib',{}).values(),default=0))
        return max(self.peak(mode,frames),resources.peak_bytes(frames,base,stage['bytes_per_output_frame']))

    def shared(self, modes):
        needed=any(m is True or (isinstance(m,str) and m == 'frax') for m in modes)
        return int(self.value['compile_extra_gib']*resources.GIB) if needed else 0

    def report(self):
        return dict(path=str(self.path), revision=self.revision, reload_interval_s=2., **copy.deepcopy(self.value))
