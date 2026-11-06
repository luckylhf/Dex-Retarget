"""Observed stage envelopes, remaining work and independently reserved compilation."""
import hashlib
import json
from .resources import GIB,MIB
from .cli import atomic_json
from .resource_events import remaining

COLD={'starting','ik_prepare','ik_compile','waiting_compile','idle'}


def envelope(samples,frames,fallback_per):
    # Fit only with enough distinct lengths and span. Every observed peak is an
    # upper-envelope constraint, not an average that can hide a difficult clip.
    per=float(fallback_per)
    xs=[v['frames'] for v in samples];ys=[v['peak_bytes'] for v in samples]
    if len(set(xs))>=3 and max(xs)-min(xs)>=500:
        center=sum(xs)/len(xs);den=sum((x-center)**2 for x in xs)
        per=max(0.,sum((x-center)*y for x,y in zip(xs,ys))/den)
        # Extrapolation remains conservative until the requested length is seen.
        if frames>max(xs):per=max(per,float(fallback_per))
    intercept=max(y-per*x for x,y in zip(xs,ys))
    return int(max(.25*GIB,intercept+per*frames)+64*MIB)


def remember(samples,frames,peak):
    # Constant-size per length bucket, avoiding full-library O(jobs^2) JSON I/O.
    for sample in samples:
        if sample['frames']//250==frames//250:
            sample['frames']=min(sample['frames'],frames)
            sample['peak_bytes']=max(sample['peak_bytes'],peak)
            return
    samples.append(dict(frames=frames,peak_bytes=peak))


class MemoryLearning:
    def __init__(self,root,identity,enabled=True):
        tag=hashlib.sha256(json.dumps(dict(schema=2,identity=identity),sort_keys=True).encode()).hexdigest()
        self.path=root/'memory_profiles'/f'{tag}.json';self.enabled=enabled
        self.profiles=json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(mode,source,cache,reuse=False):
        return '|'.join((mode,source,'reuse' if reuse else 'solve','jax' if mode!='v8' or cache in ('jax_hot','jax_imported') else 'pure'))

    def predict(self,mode,source,cache,frames,policy,stage=None,reuse=False):
        fallback=policy.peak(mode,frames)
        profile=self.profiles.get(self.key(mode,source,cache,reuse))
        if not self.enabled or not profile or profile['completed_jobs']<2:return fallback
        per=policy.value['stages'][mode]['bytes_per_output_frame']
        whole=envelope(profile['samples'],frames,per)
        if stage is None:return whole
        estimates=[]
        for phase in remaining(mode,stage):
            if phase=='loading':continue
            samples=profile['stage_samples'].get(phase,[])
            # Unseen or insufficient stages cannot release their reservation.
            estimates.append(envelope(samples,frames,per) if profile['stage_jobs'].get(phase,0)>=2 else max(whole,fallback))
        return max(estimates,default=whole)

    def observe(self,slot,item,pss):
        peaks=item.setdefault('phase_peaks',{})
        event=getattr(slot,'telemetry',{}) or {}
        phase=event.get('phase','starting');cache=event.get('cache_state','pure')
        if phase not in COLD:phase=event.get('stage',phase)
        key=phase+'|'+cache
        peaks[key]=max(peaks.get(key,0),int(pss),int(event.get('pss_bytes',0)))
        for key,value in getattr(slot,'phase_peaks',{}).items():peaks[key]=max(peaks.get(key,0),value)

    def compiler_extra(self,mode,source,policy):
        fallback=int(policy.value['compile_extra_gib']*GIB)
        profile=self.profiles.get(self.key(mode,source,'jax_hot'),{})
        if not self.enabled or profile.get('cold_jobs',0)<2:return fallback
        return max(256*MIB,profile['compile_extra_peak_bytes']+128*MIB)

    def complete(self,slot,item,response,case,frames,policy):
        if not self.enabled or response['status']!='complete':return
        peaks=item.get('phase_peaks',{});groups={}
        for key,value in peaks.items():
            phase,cache=key.split('|')
            if phase in COLD:continue
            group=self.key(item['mode'],case['source'],cache,item.get('reuse',False))
            stages=groups.setdefault(group,{})
            stages[phase]=max(stages.get(phase,0),value)
        for key,stages in groups.items():
            resident=max(stages.values(),default=0)
            if not resident:continue
            profile=self.profiles.setdefault(key,dict(completed_jobs=0,samples=[],stage_samples={},stage_jobs={},phases={}))
            profile['completed_jobs']+=1
            remember(profile['samples'],frames,resident)
            for phase,value in stages.items():
                remember(profile['stage_samples'].setdefault(phase,[]),frames,value)
                profile['stage_jobs'][phase]=profile['stage_jobs'].get(phase,0)+1
                profile['phases'][phase]=max(profile['phases'].get(phase,0),value)
            cold=max((v for k,v in peaks.items() if k.split('|')[0]=='ik_compile'),default=0)
            if cold and key.endswith('|jax'):
                profile['cold_jobs']=profile.get('cold_jobs',0)+1
                profile['compile_extra_peak_bytes']=max(profile.get('compile_extra_peak_bytes',0),cold-resident,0)
        if groups:
            self.path.parent.mkdir(parents=True,exist_ok=True);atomic_json(self.path,self.profiles)

    def report(self):return dict(enabled=self.enabled,path=str(self.path),minimum_completed_jobs=2,
        margin_mib=64,compile_margin_mib=128,policy='remaining_stage_upper_envelope_v2',profiles=self.profiles)
