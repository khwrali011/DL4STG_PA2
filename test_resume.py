"""Synthetic optimizer/RNG save-load regression test; not a performance experiment."""
import io
import torch
from core import Forecaster,video_loss,seed_all

def step(model,optim):
    x=torch.rand(1,5,1,64,64);y=torch.rand_like(x)
    mask=torch.tensor([[1.,1.,0.,0.,0.]])
    optim.zero_grad();loss=video_loss(model(x),y,mask);loss.backward();optim.step()
    return float(loss.detach())

def run():
    torch.set_num_threads(2);seed_all(0)
    m=Forecaster();o=torch.optim.Adam(m.parameters(),lr=.001)
    step(m,o);buf=io.BytesIO()
    torch.save({'weights':m.state_dict(),'optimizer':o.state_dict(),'rng':torch.get_rng_state()},buf)
    expected=step(m,o)
    buf.seek(0);c=torch.load(buf,weights_only=False)
    restored=Forecaster();op=torch.optim.Adam(restored.parameters(),lr=.001)
    restored.load_state_dict(c['weights']);op.load_state_dict(c['optimizer']);torch.set_rng_state(c['rng'])
    actual=step(restored,op)
    assert expected==actual
    assert all(torch.equal(x,y) for x,y in zip(m.parameters(),restored.parameters()))
    print('PASS: resumed synthetic Adam update exactly matches uninterrupted update (CPU).')
if __name__=='__main__':run()
