"""Conversion semantics shared by resume and initial reuse; no resource knobs."""
import os

def conversion_contract(args):
    from .identity import SOLVER_POLICY,QUALITY_POLICY
    result = dict(schema=1, fps=50., iterations=args.iterations,
        initial_iterations=args.initial_iterations,max_frames=getattr(args,'max_frames',None),
        legacy_frax=getattr(args,'legacy_frax',False),
        ik_block_frames=int(getattr(args,'ik_block_frames',None) or ('128' if hasattr(args,'manifest') else os.environ.get('DEX_FRAX_IK_BLOCK_FRAMES','128'))),
        solver_policy=SOLVER_POLICY,quality_policy=QUALITY_POLICY)
    if getattr(args,'source',None)=='smplx':
        from .smplx_cli import preparation_identity
        result['smplx_input']=preparation_identity(args)
    return result
