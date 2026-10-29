"""Staged orchestration. Two independent seed processes, one per GPU."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,json,subprocess,sys,hashlib,zipfile
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
os.chdir(HERE)

def git(*args):return subprocess.check_output(['git',*args],text=True).strip()

def commit_prediction(filename,text,message):
    path=Path('predictions')/filename;path.parent.mkdir(exist_ok=True)
    if not Path('.git').exists():
        git('init');git('config','user.name','Khawar Ali');git('config','user.email','local@example.invalid')
    if not path.exists():path.write_text(text)
    # Never silently replace an already committed pre-experiment prediction.
    tracked=subprocess.run(['git','ls-files','--error-unmatch',str(path)],capture_output=True).returncode==0
    if tracked and git('diff','HEAD','--',str(path)):
        raise RuntimeError('Committed prediction was edited. Restore its original text; do not rewrite the experiment record.')
    if git('status','--porcelain','--',str(path)):
        git('add',str(path));git('commit','-m',message)
    return git('log','-1','--format=%H','--',str(path))

def paired(script,extra):
    count=torch.cuda.device_count()
    if count<1:raise RuntimeError('Full experiments require a CUDA runtime; local checks use checks.py')
    if count>=2:
        ps=[subprocess.Popen([sys.executable,script,*extra,'--seed',str(seed),'--device',f'cuda:{seed}']) for seed in [0,1]]
        codes=[p.wait() for p in ps]
        if any(codes):raise RuntimeError(f'Workers failed: {codes}. Rerun same stage to resume.')
    else:
        for seed in [0,1]:subprocess.run([sys.executable,script,*extra,'--seed',str(seed),'--device','cuda:0'],check=True)

def evaluate_pair(name):
    for seed in [0,1]:
        if not Path(f'outputs/{name}_seed{seed}_metrics.json').exists():
            subprocess.run([sys.executable,'evaluate.py','--model',name,'--seed',str(seed),'--device',f'cuda:{seed%max(1,torch.cuda.device_count())}'],check=True)

def main(stage):
    Path('outputs').mkdir(exist_ok=True)
    if stage=='setup':
        from checks import run
        run()
        from core import Videos
        Videos('data',download=True)
        from flow import run as flow
        flow()
        print('Setup finished. Inspect Figure 1 before baseline; prediction text will include its measured median.')
    elif stage=='baseline':
        flow=json.loads(Path('outputs/flow.json').read_text());v=flow['median_speed']
        text=(f'Q2.5 - recorded before baseline transformation and speed experiments\n\n'
              f'I expect A to be imperfect under both mirror and rotation, with rotation usually harder because upright digit appearance is absent after rotation. '
              f'The measured flow median is {v:.6g} pixels/frame; 2x and 3x correspond to approximately {2*v:.6g} and {3*v:.6g}. '
              f'A has a one-step reach of 8 pixels per axis (8 sqrt(2) along a diagonal); I expect worsening at higher speeds, '
              f'especially when displacement components exceed 8 pixels, but this is a hypothesis, not a hard speed cutoff.\n')
        if Path('outputs/A_seed0_metrics.json').exists() and not Path('predictions/Q2_5.txt').exists():
            raise RuntimeError('Baseline already evaluated without prediction record. Disclose missing pre-registration; do not backdate.')
        commit_prediction('Q2_5.txt',text,'Record Q2.5 before baseline stress tests')
        paired('train.py',['--model','A']);evaluate_pair('A')
        for name in ['Blank','Copy-last']:
            if not Path(f'outputs/{name}_seed0_metrics.json').exists():
                subprocess.run([sys.executable,'evaluate.py','--model',name],check=True)
        from figures import run
        run()
        print('Inspect Figures 2-5. Then run the variants stage.')
    elif stage=='variants':
        if not Path('outputs/A_seed1_metrics.json').exists():raise RuntimeError('Run baseline first')
        text=('Q2.11 - recorded before B-D training\n\n'
              'I expect B to reduce mirror and rotation errors by exposing it to transformed digit appearances, although augmentation cannot guarantee exact consistency. '
              'I expect C to improve speed generalisation through examples of faster motion, while its one-step reach remains unchanged. '
              'I expect D to help where C remains limited by local communication, although its dilated sampling can miss fine details and none of these changes guarantees a gain.\n')
        if any(Path(f'checkpoints/{m}_seed0.pt').exists() for m in 'BCD') and not Path('predictions/Q2_11.txt').exists():
            raise RuntimeError('Variants already started without Q2.11 record. Do not backdate.')
        commit_prediction('Q2_11.txt',text,'Record Q2.11 before B-D training')
        for name in 'BCD':paired('train.py',['--model',name]);evaluate_pair(name)
    elif stage=='final':
        for name in ['symB','symD']:evaluate_pair(name)
        from figures import run
        run(final=True)
        subprocess.run([sys.executable,'report.py'],check=True)
    elif stage=='report':subprocess.run([sys.executable,'report.py'],check=True)
    elif stage=='package':
        subprocess.run([sys.executable,'report.py','--strict'],check=True)
        git('add','*.py','requirements.txt','README.md','Kaggle_Run.ipynb','.gitignore','predictions','outputs','review.json')
        if git('status','--porcelain'):git('commit','-m','Add verified Q2 results and report')
        subprocess.run(['git','bundle','create','outputs/experiment_history.bundle','--all'],check=True)
        destination=Path('DL4STG_PA2_Q2_Submission.zip')
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
            for path in HERE.rglob('*'):
                rel=path.relative_to(HERE)
                if not path.is_file() or any(p in ['.git','data','__pycache__'] for p in rel.parts) or path.suffix=='.zip':continue
                z.write(path,rel)
        print(f'Submission archive: {destination.resolve()}')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['setup','baseline','variants','final','report','package'])
    main(p.parse_args().stage)
