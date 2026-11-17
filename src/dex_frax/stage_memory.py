"""Length-aware stage permits with one fully reserved completion worker."""
from . import resources as res


class StageMemory:
    def __init__(self):
        self.protected = None
        self.bound = 0
        self.grants = 0
        self.waits = 0

    def elect(self, active, usage):
        if self.protected not in active:
            # Prefer work already holding memory over starting a new finisher.
            self.protected = max(active, key=lambda s: usage.get(s, 0), default=None)

    def running(self, active, usage, exclude=None):
        return [(max(item['grant'], item['completion_peak']) if slot is self.protected else item['grant'],
                 usage.get(slot, 0)) for slot, item in active.items() if slot is not exclude]

    def can_start(self, snapshot, reserve, active, usage, estimate, completion, used, shared):
        peak = completion if self.protected is None else estimate
        return res.can_admit(snapshot, reserve, 1, 0, self.running(active, usage), peak, used, shared)[0]

    def register(self, slot, item):
        if self.protected is None:
            self.protected = slot
        self.bound=max(self.bound,item['completion_peak'])
        item['completion_peak']=self.bound
        item['initial_grant'] = item['grant']
        item['entered_initial'] = False

    def request(self, slot, item, peak, stage, retained=None):
        if stage not in ('loading','preprocessing'):
            item['entered_initial'] = True
        if not item['entered_initial']:
            peak = max(peak, item['initial_grant'])
        elif retained is not None:
            # The worker is blocked at a safe boundary. Its previous stage
            # cannot spend unused credits; keep actual residency plus slack.
            # The protected full-lifecycle entitlement remains in running().
            item['grant']=min(item['grant'],int(retained)+128*res.MIB)
        item['completion_peak'] = max(item['completion_peak'], peak)
        self.bound=max(self.bound,item['completion_peak'])
        if 'requested_peak' not in item:
            self.waits += 1
        item['requested_peak'] = peak

    def can_grant(self, slot, item, snapshot, reserve, active, usage, shared):
        peak = item['completion_peak'] if slot is self.protected else item['requested_peak']
        return res.can_admit(snapshot, reserve, 1, 0, self.running(active, usage, slot),
                             peak, usage.get(slot, 0), shared)[0]

    def grant(self, slot, item):
        item['grant'] = item.pop('requested_peak')
        slot.grant_memory()
        self.grants += 1

    def ordered(self, active):
        waiting = [s for s in active if getattr(s, 'memory_request', None) is not None]
        return sorted(waiting, key=lambda s: (s is not self.protected,
                      getattr(s, 'wait_started', 0)))

    def report(self, active):
        item = active.get(self.protected, {})
        waiting = sum(getattr(s, 'memory_request', None) is not None for s in active)
        return dict(policy='protected_completion_stage_permits_v1',
                    completion_bound_bytes=self.bound,
                    protected_case=item.get('case'), completion_peak_bytes=item.get('completion_peak', 0),
                    waiting_workers=waiting, runnable_workers=len(active)-waiting,
                    permits_granted=self.grants, requests_seen=self.waits)
