"""Read-only local status. Matches PID identity and the selected run."""
import json,sys
from pathlib import Path
from datetime import datetime
R=Path(__file__).resolve().parent
run=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path((R/'active_run.txt').read_text().strip())
def read(name,default=None):
 try:return json.loads((run/name).read_text())
 except (OSError,ValueError):return default
manifest=read('manifest.json',{});progress=read('progress.json',{});result=read('result.json');pid=int((run/'train.pid').read_text())
alive=False
try:
 proc=Path(f'/proc/{pid}')
 cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
 parts=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
 cwd=proc.joinpath('cwd').resolve(strict=True)
 trainer=any((cwd / arg.decode()).resolve()==R/'train_duet.py' for arg in cmd if arg.endswith(b'train_duet.py'))
 alive=parts[0]!='Z' and trainer and int(parts[19])==manifest.get('pid_start_ticks')
except (OSError,ValueError):pass
checkpoints=sorted(run.glob('model_*.pt'),key=lambda x:int(x.stem.split('_')[-1]));latest=checkpoints[-1] if checkpoints else None
try:exitcode=int((run/'train.exitcode').read_text())
except (OSError,ValueError):exitcode=None
status=dict(progress,run_dir=str(run),checked_at=datetime.now().astimezone().isoformat(),process_alive=alive,pid=pid,exitcode=exitcode,
 checkpoint=str(latest) if latest else None,checkpoint_bytes=latest.stat().st_size if latest else None,result=result,connection_status='local',acceptance_passed=False)
if result:status['status']=result['status']
elif not alive:status['status']='process_missing_without_result'
print(json.dumps(status,ensure_ascii=False,indent=2))
