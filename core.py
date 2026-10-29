"""Shared definitions: all experiments and checks use these functions."""
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

GROUP = [(a, m) for m in (False, True) for a in range(4)]

def transform(x, a=0, mirror=False):
    x = torch.rot90(x, a, (-2, -1))
    return x.flip(-1) if mirror else x

def inverse(x, a=0, mirror=False):
    # (M R^a)^-1 = M R^a; rotations have inverse R^-a.
    return transform(x, a, True) if mirror else transform(x, -a)

def window(video, t=1, k=1):
    """t uses handout's one-based frame numbering; retain available targets."""
    inputs = [t - 1 + k*j for j in range(5)]
    targets = [t - 1 + k*j for j in range(5, 10) if t - 1 + k*j < len(video)]
    if inputs[-1] >= len(video) or not targets:
        raise ValueError('Window has missing inputs or no targets')
    return video[inputs], video[targets]

def video_loss(pred, target, mask):
    per_frame = (pred-target).square().mean(dim=(2,3,4))
    return ((per_frame*mask).sum(1)/mask.sum(1)).mean()

class Cell(nn.Module):
    def __init__(self, cin, hidden=64, dilation=1):
        super().__init__()
        self.gates = nn.Conv2d(cin+hidden, 4*hidden, 3,
                               padding=dilation, dilation=dilation, bias=True)
    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat([x,h],1)).chunk(4,1)
        c = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
        h = torch.sigmoid(o)*torch.tanh(c)
        return h, c

class Forecaster(nn.Module):
    def __init__(self, model='A'):
        super().__init__()
        # Fixed a priori: D uses dilation 2 in both cells, all else unchanged.
        d = 2 if model == 'D' else 1
        self.cell1 = Cell(16, dilation=d)
        self.cell2 = Cell(64, dilation=d)
        self.head = nn.Conv2d(64,16,1,bias=True)
    def forward(self, x):
        b = len(x)
        z = x.new_zeros(b,64,16,16)
        s1, s2 = (z,z), (z,z)
        predictions = []
        previous = None
        for tau in range(9):
            frame = x[:,tau] if tau < 5 else previous
            s1 = self.cell1(F.pixel_unshuffle(frame,4), s1)
            s2 = self.cell2(s1[0], s2)
            previous = torch.sigmoid(F.pixel_shuffle(self.head(s2[0]),4))
            if tau >= 4:
                predictions.append(previous)
        return torch.stack(predictions,1)

class Symmetrised(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        total = None
        for a,m in GROUP:
            p = inverse(self.base(transform(x,a,m)),a,m)
            total = p if total is None else total+p
        return total/8

class Reference(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
    def forward(self,x):
        return torch.zeros_like(x) if self.kind == 'Blank' else x[:,-1:].expand(-1,5,-1,-1,-1)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

class Videos:
    """Store uint8 frames, normalize each selected video only."""
    def __init__(self, root, download=False):
        from torchvision.datasets import MovingMNIST
        self.ds = MovingMNIST(root=root, split=None, download=download)
        if len(self.ds) != 10000: raise ValueError('Expected all 10000 videos')
    def __getitem__(self,i):
        v = self.ds[i]
        if isinstance(v,(list,tuple)): raise ValueError('Use split=None')
        v = torch.as_tensor(v)
        if v.ndim == 3: v=v[:,None]
        assert tuple(v.shape)==(20,1,64,64), v.shape
        return v.float()/255
    def __len__(self): return 10000

class TrainingWindows(torch.utils.data.Dataset):
    def __init__(self, videos, model): self.videos,self.model=videos,model
    def __len__(self): return 8000
    def __getitem__(self,i):
        if self.model in ('C','D'):
            k=random.choice([1,2,3]); t=random.randint(1,{1:14,2:8,3:2}[k])
        else: k,t=1,random.randint(1,11)
        x,y=window(self.videos[i],t,k)
        if self.model=='B':
            a,m=random.choice(GROUP); x,y=transform(x,a,m),transform(y,a,m)
        target=torch.zeros_like(x); target[:len(y)]=y
        mask=torch.zeros(5); mask[:len(y)]=1
        return x,target,mask
