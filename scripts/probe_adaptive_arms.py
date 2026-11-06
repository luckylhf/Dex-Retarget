"""Explicit synthetic regression probe; never a converted training motion."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from dex_frax.v8 import convert
from dex_frax.v8.robot import Robot
from dex_frax.v8.anatomy import configure
from dex_frax.v8.bvh import load
from dex_frax.v8.source import adapt
from dex_frax.v8.contacts import prepare
from dex_frax.contact_transition import refine as transitions
from dex_frax.adaptive_refine import inspect, pose_mask, policy, refine


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    cfg=json.loads((Path(convert.__file__).parent/'config.json').read_text())
    robot=Robot();configure(robot,cfg['anatomy'])
    t=prepare(adapt(load(args.source,fps=50.),'mocap'),robot,cfg['contact'])
    transitions(t,robot)
    data=pickle.loads(args.baseline.read_bytes());x=np.c_[data['root_pos'],data['dof_pos']].astype(float)
    if len(x)!=len(t['hips']):raise ValueError('source/baseline mismatch')
    # Smoothly rotate a shoulder branch over 100 samples; preserve joint
    # position and speed feasibility so the arm-pose detector is exercised.
    a,b=480,580;j=robot.sides['Left']['arm'][2]
    bump=np.sin(np.linspace(0,np.pi,b-a))**2
    damaged=x.copy();damaged[a:b,j]=np.clip(damaged[a:b,j]+.8*bump,robot.lower[j]+1e-5,robot.upper[j]-1e-5)
    before=inspect(robot,t,damaged);bad_before=int(pose_mask(before['errors'],50.,policy()).sum())
    if bad_before==0:raise AssertionError('probe did not trigger pose detector')
    out,report=refine(robot,t,damaged,cfg['solver'])
    after=inspect(robot,t,out);bad_after=int(pose_mask(after['errors'],50.,policy()).sum())
    record=dict(scope='synthetic smooth shoulder perturbation; no training export',
                injected_window=[a,b],joint=robot.names[j-3],amplitude_rad=.8,
                bad_frames_before=bad_before,bad_frames_after=bad_after,
                input_max_speed_excess_rad_s=float(np.maximum(abs(np.diff(damaged[:,3:],axis=0))*50-robot.speed_limits,0).max()),
                output_max_speed_excess_rad_s=float(np.maximum(abs(np.diff(out[:,3:],axis=0))*50-robot.speed_limits,0).max()),
                report=report)
    args.report.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='report'},indent=2))


if __name__=='__main__':main()
