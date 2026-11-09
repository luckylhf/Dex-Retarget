"""Sparse worker phase events; no solver parameters or trajectories are changed."""
import sys
from .resources import process_memory
import os

_sink=None
_jax_hot=False
_stage='loading'
STAGES={'frax':('loading','preprocessing','initial','fast_refinement','export_check','audit','export'),
        'v8':('loading','preprocessing','initial','support','arms','body_contacts','self_collision','export_check','audit','export')}


def remaining(mode,stage):
    order=STAGES[mode]
    return order[order.index(stage):] if stage in order else order



def set_sink(sink):
    global _sink
    _sink=sink


def phase(name,compile_possible=False,**details):
    global _jax_hot,_stage
    if name in set().union(*STAGES.values()) or name=='idle':_stage=name
    if name=='ik_solve':_jax_hot=True
    if _sink is not None:
        _sink(dict(phase=name,stage=_stage,compile_possible=compile_possible,
            cache_state='jax_hot' if _jax_hot else ('jax_imported' if 'jax' in sys.modules else 'pure'),
            pss_bytes=process_memory(os.getpid()),**details))
