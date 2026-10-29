import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Videos

def run(root='data',out='outputs'):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    ds=Videos(root);directions=[];speeds=[]
    for i in range(50):
        v=np.rint(ds[i][:,0].numpy()*255).astype(np.uint8)
        for s in range(19):
            f=cv2.calcOpticalFlowFarneback(v[s],v[s+1],None,.5,3,15,3,5,1.2,0)
            dx,dy=f[...,0],f[...,1];mag=np.hypot(dx,dy)
            mask=(v[s]>51)&(v[s+1]>51)&(mag>.5)
            directions.append(np.degrees(np.arctan2(-dy[mask],dx[mask]))%360)
            speeds.append(mag[mask])
    theta=np.concatenate(directions);speed=np.concatenate(speeds)
    if len(speed)==0:raise ValueError('No optical-flow pixels retained')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    counts,edges=np.histogram(theta,bins=np.arange(0,361,20))
    ax[0].bar(edges[:-1],counts/len(theta),width=20,align='edge',edgecolor='white')
    ax[0].axhline(1/18,color='red',ls='--',label='Uniform 1/18');ax[0].legend()
    ax[0].set(xlabel='Direction (degrees; 90 = up)',ylabel='Fraction of kept pixels',title='(a) Flow direction')
    ax[1].hist(speed,bins=40,weights=np.ones(len(speed))/len(speed))
    median=float(np.median(speed));ax[1].axvline(median,color='red',label=f'Median {median:.3g}')
    ax[1].legend();ax[1].set(xlabel='Pixels / frame',ylabel='Fraction of kept pixels',title='(b) Flow speed')
    fig.suptitle(f'Figure 1 | Training indices 0-49; 950 pairs; {len(speed):,} pixels; median {median:.3g}')
    fig.tight_layout();fig.savefig(out/'figure1.png',dpi=180);plt.close(fig)
    (out/'flow.json').write_text(json.dumps(dict(median_speed=median,kept_pixels=len(speed),pairs=950,
         direction_fractions=(counts/len(theta)).tolist()),indent=2))
if __name__=='__main__':run()
