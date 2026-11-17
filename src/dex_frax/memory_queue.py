"""Queue and reduce owned work under pressure without changing conversion semantics."""
from .resources import MIB, cgroup_floor


class MemoryQueue:
    def __init__(self, ceiling, previous=None):
        previous = previous or {}
        self.ceiling = ceiling
        self.limit = min(ceiling, max(1, previous.get('dispatch_limit', ceiling)))
        self.floors = dict(previous.get('base_floor_bytes', {}))
        self.retries = previous.get('requeued_attempts', 0)
        self.recovered = 0

    @staticmethod
    def key(mode, source):
        return mode + '|' + source

    def peak(self, mode, source, frames, per_frame, estimate):
        floor = self.floors.get(self.key(mode, source), 0)
        return max(estimate, floor + frames * per_frame if floor else 0)

    def defer(self, mode, source, frames, per_frame, peak, active_count):
        # A failed estimate must not immediately admit the same work again.
        guarded = int(peak * 1.25) + 128 * MIB
        key = self.key(mode, source)
        self.floors[key] = max(self.floors.get(key, 0), guarded - frames * per_frame)
        self.limit = max(1, min(self.limit, active_count - 1))
        self.retries += 1
        self.recovered = 0

    def complete(self):
        self.recovered += 1
        if self.recovered >= 2:
            self.limit = min(self.ceiling, self.limit + 1)
            self.recovered = 0

    def report(self):
        return dict(policy='queue_reclaim_requeue_v1', dispatch_limit=self.limit,
                    requeued_attempts=self.retries, base_floor_bytes=dict(self.floors))


def pressure(snapshot, reserve):
    """Soft pressure stops dispatch; critical pressure requires releasing memory."""
    critical = (snapshot['available'] < reserve // 2
                or snapshot['cgroup_headroom'] < cgroup_floor(snapshot))
    return critical, snapshot['available'] < reserve or critical


def victim(active, usage, protected=None):
    # Drain results first. Prefer restarting early work over discarding a nearly
    # exported motion; within a stage release the largest actual allocation.
    early = {'loading', 'preprocessing', 'initial'}
    candidates=[s for s in active if s is not protected] or list(active)
    return max(candidates, key=lambda slot: (
        getattr(slot, 'telemetry', {}).get('stage', 'initial') in early,
        usage.get(slot, 0), getattr(slot, 'started', 0)), default=None)
