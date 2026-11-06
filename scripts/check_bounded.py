"""Run one development verification under a five-minute owned-process budget."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd:p.error('A command is required')
a.log.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic();timed_out=False
with a.log.open('w') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=child.wait(timeout=295.)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=5.)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        code=124
receipt=dict(command=cmd,returncode=code,timed_out=timed_out,elapsed_s=time.monotonic()-started,budget_s=300.)
a.log.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(code)
