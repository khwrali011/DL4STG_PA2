import json
from pathlib import Path
import torch
from core import *

def run(out='outputs'):
    torch.set_num_threads(2)
    torch.manual_seed(0); x=torch.rand(20,1,64,64)
    rows=[]
    for label,actual,expected in [
        ('a', transform(transform(x,0,True),0,True),x),
        ('b', transform(x,4),x),
        ('c', transform(transform(transform(x,0,True),1),0,True),transform(x,-1))]:
        diff=(actual-expected).abs().max().item(); eq=torch.equal(actual,expected)
        assert diff==0 and eq
        rows.append(dict(check=label,expected='0; torch.equal=True',yours=f'{diff}; torch.equal={eq}'))
    p=torch.zeros_like(x);p[7,0,10,20]=1
    for label,v,expected in [('d',transform(p,0,True),(7,10,43)),('e',transform(p,1),(7,43,10))]:
        idx=v.nonzero()[0]; got=(int(idx[0]),int(idx[2]),int(idx[3]));assert got==expected
        rows.append(dict(check=label,expected=str(expected),yours=str(got)))
    f=torch.arange(1,21).view(20,1,1,1).expand(20,1,64,64)
    for label,k,expected in [('f',1,([1,2,3,4,5],[6,7,8,9,10])),('g',2,([1,3,5,7,9],[11,13,15,17,19])),('h',3,([1,4,7,10,13],[16,19]))]:
        a,b=window(f,1,k);got=(a[:,0,0,0].tolist(),b[:,0,0,0].tolist());assert got==expected
        rows.append(dict(check=label,expected=str(expected),yours=str(got)))
    for a,m in GROUP: assert torch.equal(inverse(transform(x,a,m),a,m),x)
    for k,limit in [(1,14),(2,8),(3,2)]:
        for t in range(1,limit+1):
            a,b=window(x,t,k);assert len(a)==5 and len(b)>=2
        assert len(window(x,limit+1,k)[1])<2
    # Verify equal weighting of videos with two and five targets.
    pred=torch.zeros(2,5,1,1,1); target=torch.ones_like(pred);target[1]*=2
    mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,1]])
    assert video_loss(pred,target,mask).item()==2.5
    for name in ['A','D']:
        model=Forecaster(name); count=sum(p.numel() for p in model.parameters())
        assert count==480784
        v=torch.rand(1,5,1,64,64,requires_grad=True)
        y=model(v);assert y.shape==(1,5,1,64,64)
        y[:,-1].mean().backward();assert v.grad.abs().sum()>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    class Asymmetric(torch.nn.Module):
        def forward(self,v): return torch.roll(v,1,-1)*.7+torch.flip(v,[-2])*.3
    wrapped=Symmetrised(Asymmetric());v=torch.rand(2,5,1,8,8)
    gaps={}
    for label,a,m in [('M',0,True),('R',1,False)]:
        gap=(wrapped(transform(v,a,m))-transform(wrapped(v),a,m)).square().mean().item()
        assert gap<1e-12;gaps[label]=gap
    Path(out).mkdir(exist_ok=True,parents=True)
    Path(out,'table1.json').write_text(json.dumps(rows,indent=2))
    result={'status':'passed','A_parameters':480784,'D_parameters':480784,
            'A_reach_pixels_Linf':8,'D_reach_pixels_Linf':16,
            'synthetic_symmetrisation_gaps':gaps,
            'scope':'CPU correctness checks; not trained model performance'}
    Path(out,'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
