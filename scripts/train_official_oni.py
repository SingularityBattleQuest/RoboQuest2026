"""Reproducibly learn the fixed official oni gait from a geometric teacher.
The published ONNX is used for every joint action at inference; the teacher
is used only here to create training labels, never by the competition oni.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from competition.arena import gait_action
from competition.rules import VERSION

class LearnedGait(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network=torch.nn.Sequential(torch.nn.Linear(5,128),torch.nn.Tanh(),
            torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,12),torch.nn.Tanh())
    def forward(self,obs):
        return self.network(torch.cat([obs[:,:3],obs[:,45:47]],dim=1))

def train(destination='competition/oni',updates=16000):
    torch.set_num_threads(1);torch.manual_seed(2026)
    rng=np.random.default_rng(2026)
    observations=[];targets=[]
    for command in [(0,0,0),(.3,0,0),(.4,0,0),(.5,0,0),
                    (0,0,.5),(0,0,-.5),(.3,0,.5),(.3,0,-.5),
                    (.4,0,.5),(.4,0,-.5),(.5,0,.5),(.5,0,-.5)]:
        for phase in np.linspace(0,1,2000,endpoint=False):
            obs=np.zeros(47,dtype=np.float32);obs[:3]=command
            obs[-2:]=[math.sin(2*math.pi*phase),math.cos(2*math.pi*phase)]
            observations.append(obs);targets.append(gait_action(phase/1.5,command[0],command[2]))
    x=torch.tensor(np.asarray(observations));y=torch.tensor(np.asarray(targets))
    model=LearnedGait();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    for update in range(updates):
        if update==12000:
            for group in optimizer.param_groups:group['lr']=1e-4
        idx=torch.tensor(rng.integers(0,len(x),512));loss=(model(x[idx])-y[idx]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (update+1)%2000==0:print(update+1,float(loss.detach()),flush=True)
    folder=Path(destination);folder.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),folder/'training_weights.pt')
    torch.onnx.export(model,torch.zeros(1,47),str(folder/'walk.onnx'),
                      input_names=['obs'],output_names=['action'],opset_version=17,dynamo=False)
    from competition.submission import Policy
    policy=Policy((folder/'walk.onnx').read_bytes(),47,12)
    with torch.no_grad():pred=model(x).numpy()
    error=float(np.max(np.abs(pred-y.numpy())))
    manifest=dict(version=VERSION,model='walk.onnx',sha256=hashlib.sha256((folder/'walk.onnx').read_bytes()).hexdigest(),
        observation_dim=47,action_dim=12,normalization='identity embedded in trained input contract',
        training=dict(recipe='scripts.train_official_oni.train',seed=2026,algorithm='supervised behavior cloning',
            samples=len(x),updates=updates,framework=f'PyTorch {torch.__version__}; CPU',teacher_max_action_error=error,
            used_observations='velocity commands and public phase; physical feedback features reserved'))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':train()
