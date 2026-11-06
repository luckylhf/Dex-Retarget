"""Build a private, hash-pinned multi-motion corpus from authorized local BVHs.

Excerpts are standalone fixtures, not equivalent windows of full-clip fitting.
The full source copies, selection evidence and outputs stay in regression_local.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from dex_frax.v8.bvh import load

CASES = [
    ('walk_turn','walk1_subject1','turn','walking and heading change'),
    ('run','run1_subject2','speed','running'),
    ('sprint','sprint1_subject2','speed','fast locomotion'),
    ('dance','dance1_subject2','hands','coordinated upper body'),
    ('jump','jumps1_subject2','vertical','jump or landing transition'),
    ('obstacle','obstacles1_subject2','vertical','stepping over obstacles'),
    ('aiming','aiming1_subject1','straight','near-straight elbow orientation'),
    ('fight','fight1_subject2','hands','rapid arm movement'),
    ('ground','ground2_subject2','low','low body and possible body contact'),
    ('get_up','fallAndGetUp2_subject2','rise','fall/get-up transition'),
    ('stumble','pushAndStumble1_subject2','turn','disturbed balance and turning'),
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lafan',type=Path,required=True)
    parser.add_argument('--nobody',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    sources=root/'sources';clips=root/'clips';sources.mkdir();clips.mkdir();rows=[]
    for ident,stem,criterion,purpose in CASES:
        src=args.lafan/(stem+'.bvh');full=sources/src.name;shutil.copy2(src,full)
        motion=load(full);p,r=motion.fk();fps=motion.fps;n=len(motion.local_p)
        hip=p['Hips'];speed=np.linalg.norm(np.gradient(hip,1/fps,axis=0),axis=1)
        hands=sum(np.linalg.norm(np.gradient(p[s+'Hand'],1/fps,axis=0),axis=1) for s in ('Left','Right'))
        lateral=p['LeftUpLeg']-p['RightUpLeg'];yaw=np.unwrap(np.arctan2(lateral[:,1],lateral[:,0]))
        turn=abs(np.gradient(yaw,1/fps));vertical=abs(np.gradient(hip[:,2],1/fps))
        straight=np.zeros(n)
        for s in ('Left','Right'):
            u=p[s+'ForeArm']-p[s+'Arm'];v=p[s+'Hand']-p[s+'ForeArm']
            straight+=np.sum(u*v,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1))
        raw={'speed':speed,'hands':hands,'turn':turn,'vertical':vertical,
             'straight':straight,'low':-hip[:,2],'rise':np.gradient(hip[:,2],1/fps)}[criterion]
        width=min(n,round(6.*fps)+1)
        # Select the strongest two-second event, then include six seconds of context.
        smooth=np.convolve(raw,np.ones(max(1,round(2*fps)))/max(1,round(2*fps)),mode='same')
        half=width//2;center=int(np.argmax(smooth[half:n-half])+half) if n>width else half
        a=max(0,min(n-width,center-half));b=a+width
        text=full.read_text();head,body=text.split('MOTION',1);lines=body.strip().splitlines()
        clip=clips/(ident+'.bvh')
        clip.write_text(head+'MOTION\nFrames: '+str(width)+'\n'+lines[1]+'\n'+'\n'.join(lines[2+a:2+b])+'\n')
        rows.append(dict(id=ident,source='lafan',purpose=purpose,selection=criterion,
            selection_score=float(smooth[center]),original_path=str(src.resolve()),
            full_source=str(full.relative_to(root)),full_source_sha256=sha(full),
            input=str(clip.relative_to(root)),input_sha256=sha(clip),
            source_frame_start=a,source_frame_end_exclusive=b,source_fps=fps,
            source_seconds=[a/fps,(b-1)/fps],fixture_duration_s=(width-1)/fps,
            hip_height_range_m=[float(hip[a:b,2].min()),float(hip[a:b,2].max())],
            peak_hand_speed_m_s=float(hands[a:b].max()),output_fps=50.))
        print(ident,a,b,flush=True)
    full=sources/'nobody.bvh';shutil.copy2(args.nobody,full);clip=clips/'nobody.bvh';shutil.copy2(full,clip)
    rows.insert(0,dict(id='nobody',source='mocap',purpose='full historical regression anchor',selection='entire_clip',
        original_path=str(args.nobody.resolve()),full_source=str(full.relative_to(root)),full_source_sha256=sha(full),
        input=str(clip.relative_to(root)),input_sha256=sha(clip),output_fps=50.))
    manifest=dict(schema=1,private=True,publication='excluded by /regression_local/ in .gitignore',
        scope='one full nobody and eleven six-second LAFAN standalone excerpts; no moving SOMA sample available',
        classification='filename categories plus measured kinematic selection; not manual semantic or force labels',cases=rows)
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(root/'manifest.json')


if __name__=='__main__':main()
