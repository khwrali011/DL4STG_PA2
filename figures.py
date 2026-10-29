import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from core import Videos,window,transform
from evaluate import load_model

def grid(rows,labels,path,title):
    n=max(len(r) for r in rows)
    fig,axes=plt.subplots(len(rows),n,figsize=(n*1.05,len(rows)*1.22),squeeze=False)
    for i,row in enumerate(rows):
        for j in range(n):
            ax=axes[i,j];ax.set_xticks([]);ax.set_yticks([])
            if j<len(row):ax.imshow(row[j].squeeze(),cmap='gray',vmin=0,vmax=1)
            else:ax.set_visible(False)
            if j==0:ax.set_ylabel(labels[i],fontsize=8)
            if i==0:ax.set_title(f'{j+1}',fontsize=8)
    fig.suptitle(title,fontsize=10);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def gif(rows,path,title):
    frames=[]
    # Keep full input strip fixed, update the forecast column over h=1,...,5.
    for h in range(5):
        canvas=Image.new('RGB',(6*96+140,4*96+38),'white');draw=ImageDraw.Draw(canvas)
        draw.text((5,5),title+f' | horizon {h+1}',fill='black')
        for i,row in enumerate(rows):
            draw.text((5,40+i*96),['X / f(X)','TX / TY','TX / f(TX)','TX / Tf(X)'][i],fill='black')
            for j,frame in enumerate(list(row[:5])+[row[5+h]]):
                img=Image.fromarray(np.uint8(np.clip(frame.squeeze(),0,1)*255)).resize((96,96))
                canvas.paste(img,(140+j*96,38+i*96))
        frames.append(canvas)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=450,loop=0)

def major_components(frame):
    import cv2
    count,labels,stats,cent=cv2.connectedComponentsWithStats(np.uint8(frame>.2),connectivity=8)
    return int(sum(stats[1:,cv2.CC_STAT_AREA]>=20))

def choose(ds,out):
    file=out/'selection.json'
    if file.exists():return json.loads(file.read_text())
    a=np.load(out/'A_seed0_per_video.npz')
    ids={k:[9000+int(j) for j in np.argsort(-a[k],kind='stable')[:2]] for k in ['CM','CR']}
    speed=None;overlap=None
    for i in range(9000,10000):
        v=ds[i][:,0].numpy();counts=[major_components(f) for f in v]
        if speed is None and all(c==2 for c in counts):speed=i
        if overlap is None and counts[0]==2 and any(c==1 for c in counts[5:10]):overlap=i
        if speed is not None and overlap is not None:break
    if speed is None or overlap is None:raise RuntimeError('Choose overlap and separated test videos manually in selection.json')
    others=[i for i in [9000,9001,9002,9003] if i!=overlap][:2]
    ids.update(figure2=[overlap]+others,speed=speed,reviewed=False,
        note='Overlap and separation selected by connected-component heuristic; visually verify and set reviewed=true.')
    file.write_text(json.dumps(ids,indent=2));return ids

@torch.no_grad()
def run(root='data',ckpt='checkpoints',out='outputs',device='cuda:0',final=False):
    out=Path(out);ds=Videos(root);sel=choose(ds,out)
    for key in ['CM','CR','figure2']:
        assert all(9000<=i<10000 for i in sel[key])
    assert 9000<=sel['speed']<10000
    a=np.load(out/'A_seed0_per_video.npz')
    if not final:
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        for ax,key in zip(axes,['CM','CR']):
            ax.hist(a[key],bins=40)
            for idx in sel[key]:ax.axvline(a[key][idx-9000],label=f'Video {idx}',ls='--')
            ax.set(xlabel=key,ylabel='Test video count');ax.legend()
        fig.suptitle('Figure 4 | Model A, seed 0, 1x, t=1');fig.tight_layout()
        fig.savefig(out/'figure4.png',dpi=180);plt.close(fig)
    m,_=load_model('symD' if final else 'A',0,ckpt,device)
    def pred(x):return m(x[None].to(device))[0].cpu().numpy()
    if not final:
        for idx in sel['figure2']:
            x,y=window(ds[idx]);p=pred(x)
            grid([x.numpy(),p,y.numpy()],['Input','Prediction','Ground truth'],out/f'figure2_video{idx}.png',f'Figure 2 | A seed 0 | Test video {idx}, 1x')
    for key,aa,mir in [('CM',0,True),('CR',1,False)]:
        for rank,idx in enumerate(sel[key],1):
            x,y=window(ds[idx]);p=pred(x);tx=transform(x,aa,mir)
            rows=[np.concatenate([x.numpy(),p]),np.concatenate([tx.numpy(),transform(y,aa,mir).numpy()]),
                  np.concatenate([tx.numpy(),pred(tx)]),
                  np.concatenate([tx.numpy(),transform(torch.from_numpy(p),aa,mir).numpy()])]
            stem=f'figure{6 if final else 3}{"a" if final else ""}_{key}_{rank}_video{idx}'
            title=f'Figure {"6(a)" if final else "3"} | {"symD" if final else "A"} seed 0 | {key} | video {idx}'
            grid(rows,['X / f(X)','TX / TY','TX / f(TX)','TX / Tf(X)'],out/(stem+'.png'),title)
            gif(rows,out/(stem+'.gif'),title)
    rows=[];labels=[]
    for k in [1,2,3]:
        x,y=window(ds[sel['speed']],1,k);p=pred(x)
        rows += [np.concatenate([x.numpy(),p[:2]]),np.concatenate([x.numpy(),y[:2].numpy()])]
        labels += [f'{k}x prediction',f'{k}x truth']
    grid(rows,labels,out/('figure6b.png' if final else 'figure5.png'),
         f'Figure {"6(b)" if final else "5"} | {"symD" if final else "A"} seed 0 | video {sel["speed"]} | 5 inputs + 2 futures')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--device',default='cuda:0')
    a=p.parse_args();run(device=a.device,final=a.final)
