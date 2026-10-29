import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from core import *

def load_model(name,seed,ckpt,device):
    if name in ['Blank','Copy-last']:return Reference(name).to(device).eval(),None
    base=name.replace('sym','')
    ck=torch.load(Path(ckpt)/f'{base}_seed{seed}.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==30,'Only final-epoch models are valid'
    m=Forecaster(base);m.load_state_dict(ck['weights'])
    if name.startswith('sym'):m=Symmetrised(m)
    return m.to(device).eval(),ck

def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)

def mse(p,y):return (p-y).square().mean((2,3,4))

@torch.no_grad()
def evaluate(name,seed,root='data',ckpt='checkpoints',out='outputs',device='cuda:0'):
    seed_all(seed);torch.set_num_threads(2)
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    m,ck=load_model(name,seed,ckpt,device);data=Videos(root)
    values={};elapsed=0
    # Timing starts after each input batch reaches device, as required.
    # Warm-up uses validation index 8000, with no performance reporting.
    warm,_=window(data[8000]);m(warm[None].to(device));sync(device)
    for k in [1,2,3]:
        for begin in range(9000,10000,32):
            pairs=[window(data[i],1,k) for i in range(begin,min(begin+32,10000))]
            x=torch.stack([p[0] for p in pairs]).to(device)
            y=torch.stack([p[1] for p in pairs]).to(device)
            sync(device);tic=time.perf_counter();pred=m(x);sync(device)
            if k==1:elapsed+=time.perf_counter()-tic
            values.setdefault(f'mse_k{k}',[]).append(mse(pred[:,:y.shape[1]],y).cpu().numpy())
            if k==1:
                for label,a,mir in [('M',0,True),('R',1,False)]:
                    tp=m(transform(x,a,mir))
                    values.setdefault('C'+label,[]).append(mse(tp,transform(pred,a,mir)).mean(1).cpu().numpy())
                    values.setdefault('E'+label,[]).append(mse(tp,transform(y,a,mir)).mean(1).cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in values.items()}
    assert all(len(v)==1000 for v in arrays.values())
    np.savez_compressed(out/f'{name}_seed{seed}_per_video.npz',**arrays)
    result=dict(model=name,seed=seed,parameters=sum(p.numel() for p in m.parameters()),
        E5_1=float(arrays['mse_k1'].mean()),E5_2=float(arrays['mse_k2'].mean()),
        E2_1=float(arrays['mse_k1'][:,:2].mean()),E2_2=float(arrays['mse_k2'][:,:2].mean()),
        E2_3=float(arrays['mse_k3'][:,:2].mean()),
        train_seconds=float(np.mean(ck['epoch_seconds'])) if ck else 0.,inference_seconds=elapsed,
        hardware=ck['hardware'] if ck else str(device),
        **{f'MSE{h+1}':float(arrays['mse_k1'][:,h].mean()) for h in range(5)},
        **{key:float(arrays[key].mean()) for key in ['CM','EM','CR','ER']})
    (out/f'{name}_seed{seed}_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--root',default='data');p.add_argument('--ckpt',default='checkpoints');p.add_argument('--out',default='outputs')
    p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(a.model,a.seed,a.root,a.ckpt,a.out,a.device)
