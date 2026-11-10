"""Conversion semantics shared by resume and initial reuse; no resource knobs."""
import os

def conversion_contract(args):
    from .identity import SOLVER_POLICY,QUALITY_POLICY
    from .dispatch import use_unified
    result = dict(schema=6,bvh_units=getattr(args,'bvh_units','cm'),bvh_translations=getattr(args,'bvh_translations','absolute'),allow_local_slowdown=bool(getattr(args,'allow_local_slowdown',False)),retiming_policy='prepared_target_local_root_slowdown_v2',refine_initial_policy='pelvis_z_feet_feasibility_waist_feet_arms_v6', profile=getattr(args,'profile',None), repair_budget=getattr(args,'repair_budget',None), arm_principles='whole_arm_resemblance_v2', repair_similarity=getattr(args,'repair_similarity','soft'),pipeline=getattr(args,'pipeline','unified'),
        unified_repair=getattr(args,'unified_repair',use_unified(args)), fps=50., iterations=args.iterations,
        initial_iterations=args.initial_iterations,max_frames=getattr(args,'max_frames',None),
        legacy_frax=True if getattr(args,'profile',None) in ('fast','refine') else getattr(args,'legacy_frax',False),
        ik_block_frames=int(getattr(args,'ik_block_frames',None) or ('128' if hasattr(args,'manifest') else os.environ.get('DEX_FRAX_IK_BLOCK_FRAMES','128'))),
        solver_policy=SOLVER_POLICY,quality_policy=QUALITY_POLICY)
    if getattr(args,'source',None)=='smplx':
        from .smplx_cli import preparation_identity
        result['smplx_input']=preparation_identity(args)
    return result
