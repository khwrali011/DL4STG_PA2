import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from core import Forecaster,Videos,TrainingWindows,video_loss,seed_all

def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def run(model_name,seed,root,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/f'{model_name}_seed{seed}.pt'
    seed_all(seed);torch.set_num_threads(2)
    model=Forecaster(model_name).to(device)
    optim=torch.optim.Adam(model.parameters(),lr=1e-3)
    data=TrainingWindows(Videos(root),model_name)
    generator=torch.Generator().manual_seed(seed)
    loader=torch.utils.data.DataLoader(data,batch_size=32,shuffle=True,num_workers=0,generator=generator)
    start=0;times=[];losses=[]
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['model']==model_name and ck['seed']==seed and ck['protocol']=='q2-v1-fp32'
        model.load_state_dict(ck['weights']);optim.load_state_dict(ck['optimizer'])
        start=ck['epoch'];times=ck['epoch_seconds'];losses=ck['losses']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng']);generator.set_state(ck['loader_rng'])
        if str(device).startswith('cuda'):torch.cuda.set_rng_state(ck['cuda_rng'],device)
    print(f'{model_name} seed {seed}: resume {start}/30 on {device}',flush=True)
    for epoch in range(start,30):
        model.train()
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        tic=time.perf_counter();total=0
        for x,y,mask in loader:
            x,y,mask=x.to(device),y.to(device),mask.to(device)
            optim.zero_grad(set_to_none=True)
            loss=video_loss(model(x),y,mask)
            loss.backward();optim.step();total+=float(loss.detach())*len(x)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        times.append(time.perf_counter()-tic);losses.append(total/8000)
        hardware=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU'
        atomic_save(dict(protocol='q2-v1-fp32',model=model_name,seed=seed,epoch=epoch+1,
            weights=model.state_dict(),optimizer=optim.state_dict(),epoch_seconds=times,losses=losses,
            hardware=hardware,torch_version=torch.__version__,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            loader_rng=generator.get_state(),cuda_rng=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None),path)
        print(f'{model_name} seed {seed} epoch {epoch+1}/30 MSE={losses[-1]:.6g} seconds={times[-1]:.1f}',flush=True)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list('ABCD'),required=True)
    p.add_argument('--seed',type=int,choices=[0,1],required=True);p.add_argument('--root',default='data')
    p.add_argument('--out',default='checkpoints');p.add_argument('--device',default='cuda:0')
    a=p.parse_args();run(a.model,a.seed,a.root,a.out,a.device)
