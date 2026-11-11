"""Fair, dependency-aware work with context expansion and numerical checkpoints."""
import hashlib
import json
import numpy as np

RECIPES={'collision':['direct10','recover10','direct20','recover20','direct30','recover30'],
         'torso':['source_fit','source_fit_collision','source_branch_collision'],
         'arms':['fit','current_best','guided_current','original_frax','guided_original'],
         'support':['geometry5_elastic','geometry10_elastic'],'body':['solve','solve_more'],
         'leg_collision':['legs5','legs10'],
         'dense_floor':['solve','solve_more'],'jitter':['temporal','temporal_more']}
for tier in (10,20,30):
    RECIPES['collision'].insert(RECIPES['collision'].index('direct'+str(tier)), 'sweep'+str(tier))
    anchor=RECIPES['collision'].index('recover'+str(tier))+1
    RECIPES['collision'][anchor:anchor]=['continuous'+str(tier)+'_'+axis for axis in ('zero','xpos','xneg','ypos','yneg','zpos','zneg')]
# Strict stage ordering; visits only rotate windows within the same stage.
PHASE={'support':0,'body':0,'dense_floor':0,'jitter':0,'torso':1,'body_collision':1,'leg_collision':2,'collision':3,'arms':4}
PRIORITY={'support':0,'body':1,'dense_floor':2,'jitter':3,'torso':4,'body_collision':5,'leg_collision':6,'collision':7,'arms':8}
PADDING=(20,50,100)


def jobs_from_masks(masks,n,core=101,padding=20):
    jobs=[]
    for stage,side,mask in masks:
        local_core=11 if stage in ('dense_floor','jitter') else 21 if stage in ('support','body','leg_collision') else core
        local_padding=5 if stage in ('dense_floor','jitter','support','body','leg_collision') else padding
        ranges=[(a,min(n,a+local_core)) for a in range(0,n,local_core)]
        if stage=='collision':
            # Keep a bounded collision episode away from immutable joins.
            # Arbitrary grid cuts can freeze already-penetrating boundary poses.
            edges=np.diff(np.r_[False,mask,False].astype(int));merged=[]
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                if merged and a-merged[-1][1]<=2*local_padding+4 and b-merged[-1][0]<=201:merged[-1]=(merged[-1][0],int(b))
                else:merged.append((int(a),int(b)))
            ranges=[]
            for a,b in merged:
                ranges.extend([(a,b)] if b-a<=201 else [(i,min(b,i+core)) for i in range(a,b,core)])
        for start,end in ranges:
            if not np.any(mask[start:end]):continue
            a=max(0,start-local_padding);b=min(n,end+local_padding)
            if b-a<5:continue
            jobs.append(dict(module=stage,side=side,start=a,end=b,core_start=start,core_end=end,
                             severity=int(np.count_nonzero(mask[start:end]))))
    return sorted(jobs,key=lambda v:(PRIORITY[v['module']],-v['severity'],v['start'],v['side'] or ''))


class WorkQueue:
    def __init__(self,states=None,retry_timed=False):
        self.states={} if states is None else states
        if retry_timed:
            for state in self.states.values():
                # Preserve numerical progress, and try fresh families before retries.
                state['stalled']=[]
    @staticmethod
    def key(job):return '|'.join(str(job[k]) for k in ('module','side','core_start','core_end'))
    def state(self,job):
        return self.states.setdefault(self.key(job),dict(cursor=0,visits=0,reactivations=0,level=0,done=[],timed_recipes=[],stalled=[],progress={},seen={}))
    def choices(self,job):
        state=self.state(job);recipes=job.get('candidate_order',RECIPES.get(job['module'],[]))
        fresh=[r for r in recipes if r not in state['done'] and r not in state['timed_recipes']]
        retry=[r for r in state['timed_recipes'] if r not in state['stalled']]
        counts=state.get('family_visits',{})
        order={r:i for i,r in enumerate(recipes)}
        fresh.sort(key=lambda r:(counts.get(r,0),order[r]))
        return fresh+retry
    def configure(self,jobs,rows):
        """Only real changed context invalidates previous outcomes; no retry cap."""
        out=[];n=len(rows)
        for raw in jobs:
            job=dict(raw);state=self.state(job)
            if job['module'] not in RECIPES:out.append(job);continue
            # A small gain must not keep a failing seam trapped forever in
            # the same narrow context. Finish the candidate families, then
            # expand before repeating a productive round at maximum context.
            if not self.choices(job):
                if state['level']+1<len(PADDING):
                    state['level']+=1;state['context']=None
                    state.update(done=[],timed_recipes=[],stalled=[],progress={},round_improved=False)
                    state.pop('own_result_context',None)
                elif state.get('round_improved'):
                    state.update(done=[],timed_recipes=[],stalled=[],progress={},round_improved=False)
                    state['rounds']=state.get('rounds',0)+1
            padding=min(10,5+5*state['level']) if job['module'] in ('dense_floor','jitter','support','body','leg_collision') else PADDING[state['level']];job.update(start=max(0,job['core_start']-padding),end=min(n,job['core_end']+padding),context_level=state['level'])
            context=hashlib.sha256(np.ascontiguousarray(rows[max(0,job['start']-1):min(n,job['end']+1)]).tobytes()).hexdigest()
            if state.get('context')!=context:
                own=context==state.pop('own_result_context',None)
                if state.get('context') is not None and not own:state['reactivations']+=1
                completed=list(state['done']) if own else list(state['seen'].get(context,[]))
                state['context']=context;state['done']=completed
                state['timed_recipes']=[];state['stalled']=[];state['progress']={};state['cursor']=len(state['done'])
            out.append(job)
        return out
    def actionable(self,jobs):return [j for j in jobs if self.choices(j)]
    def next(self,jobs):
        available=self.actionable(jobs)
        if not available:return None
        return min(available,key=lambda j:(PHASE[j['module']],self.state(j)['visits'],PRIORITY[j['module']],-j['severity'],j['start']))
    def recipe(self,job):return self.choices(job)[0]
    def progress(self,job):return self.state(job)['progress'].setdefault(self.recipe(job),{})
    def complete(self,job,kept,timed=False,recipe=None,result_rows=None):
        state=self.state(job);recipe=recipe or self.recipe(job);state['visits']+=1
        counts=state.setdefault('family_visits',{});counts[recipe]=counts.get(recipe,0)+1
        if timed:
            if recipe not in state['timed_recipes']:state['timed_recipes'].append(recipe)
            value=hashlib.sha256(json.dumps(state['progress'].get(recipe,{}),sort_keys=True).encode()).hexdigest()
            old=state.setdefault('progress_hash',{}).get(recipe)
            stalls=state.setdefault('no_progress',{});stalls[recipe]=stalls.get(recipe,0)+1 if old==value else 0
            state['progress_hash'][recipe]=value
            if stalls[recipe]>=2:state['stalled'].append(recipe)
        else:
            if recipe not in state['done']:state['done'].append(recipe)
            state['timed_recipes']=[r for r in state['timed_recipes'] if r!=recipe]
            state['progress'].pop(recipe,None)
            if state.get('context'):state['seen'][state['context']]=list(state['done'])
        if kept:
            state['round_improved']=True
            if result_rows is not None:
                lo=max(0,job['start']-1);hi=min(len(result_rows),job['end']+1)
                state['own_result_context']=hashlib.sha256(np.ascontiguousarray(result_rows[lo:hi]).tobytes()).hexdigest()
        state['cursor']=len(state['done'])
    def pending(self,jobs):
        result=[]
        for job in jobs:
            state=self.state(job);choices=self.choices(job)
            reason=('unsupported_collision_class' if job['module'] not in RECIPES else
                'untried_candidates' if choices else 'time_sliced_candidates' if state['timed_recipes'] else 'candidate_families_exhausted')
            result.append(dict(job,next_recipe=choices[0] if choices else None,reason=reason,
                time_sliced_recipes=state['timed_recipes'],context_level=state['level']))
        return result


def time_sliced(error):
    """Pose/size budgets are failed constraints, never elapsed-time signals."""
    text=str(error).lower()
    return (isinstance(error,TimeoutError) or any(k in text for k in ('deadline','watchdog','time limit'))
            or text in {'shoulder_recovery_budget','continuous_shoulder_seed_budget'})
