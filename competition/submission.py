"""Strict ONNX-only submissions: no participant Python, pickle or external files."""
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import onnx
import onnxruntime as ort
from competition.rules import VERSION

MAX_BYTES = 100*1024*1024
ALLOWED_OPS = {'Gemm','MatMul','Add','Sub','Mul','Div','Sqrt','Clip','Tanh','Relu',
               'Sigmoid','Identity','Constant','Cast','Reshape','Flatten','Concat',
               'Gather','Shape','Unsqueeze','Squeeze','Expand','Slice','Neg','Abs',
               'Min','Max','ReduceMean','ReduceSum','Pow','Where','Greater','Less',
               'Equal','Sin','Cos','Atan','Softmax','ConstantOfShape','Exp'}


class SubmissionError(ValueError):
    pass


def graph_check(data, inputs, outputs):
    model = onnx.load_model_from_string(data)
    if model.functions or len(model.graph.node)>2000:
        raise SubmissionError('Model functions or more than 2000 nodes are not permitted')
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            raise SubmissionError('External tensor data is forbidden')
        if np.prod(tensor.dims,dtype=np.float64)>16_000_000:
            raise SubmissionError('Tensor exceeds size limit')
    for node in model.graph.node:
        if node.domain not in ('','ai.onnx') or node.op_type not in ALLOWED_OPS:
            raise SubmissionError(f'Unsupported ONNX operation: {node.domain}:{node.op_type}')
        if any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS) for a in node.attribute):
            raise SubmissionError('Nested graphs are forbidden')
        for a in node.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_location == onnx.TensorProto.EXTERNAL:
                raise SubmissionError('External constant forbidden')
    onnx.checker.check_model(model)
    if len(model.graph.input)!=1 or len(model.graph.output)!=1:
        raise SubmissionError('Exactly one observation input and one action output required')
    for tensor, size in [(model.graph.input[0],inputs),(model.graph.output[0],outputs)]:
        t=tensor.type.tensor_type
        shape=t.shape.dim
        if t.elem_type != onnx.TensorProto.FLOAT or len(shape)!=2 or shape[1].dim_value!=size:
            raise SubmissionError(f'Expected float32 [batch,{size}]')
        if shape[0].HasField('dim_value') and shape[0].dim_value!=1:
            raise SubmissionError('Batch size must permit 1')


class Policy:
    def __init__(self,data,obs_dim,action_dim):
        graph_check(data,obs_dim,action_dim)
        options=ort.SessionOptions()
        options.intra_op_num_threads=1; options.inter_op_num_threads=1
        options.enable_cpu_mem_arena=False
        self.session=ort.InferenceSession(data,sess_options=options,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name
        self.obs_dim,self.action_dim=obs_dim,action_dim
        # Initial graph preparation is excluded from the per-inference budget.
        self._warmup=True
        self(np.zeros(obs_dim,dtype=np.float32))
        self._warmup=False

    def __call__(self,obs):
        obs=np.asarray(obs,dtype=np.float32)
        if obs.shape!=(self.obs_dim,) or not np.all(np.isfinite(obs)):
            raise SubmissionError('Invalid observation')
        start=time.monotonic()
        try:
            out=self.session.run(None,{self.input:obs[None]})[0]
        except Exception as error:
            raise SubmissionError(f'Inference failed: {type(error).__name__}') from error
        if not self._warmup and time.monotonic()-start>.1:
            raise SubmissionError('Inference exceeded 100ms')
        if out.shape!=(1,self.action_dim) or not np.all(np.isfinite(out)):
            raise SubmissionError('Invalid action: shape or NaN/Inf')
        if np.any(np.abs(out)>1.00001):
            raise SubmissionError('Actions must be within [-1,1]')
        return np.clip(out[0],-1,1)


class Submission:
    def __init__(self,path):
        path=Path(path)
        if path.stat().st_size>MAX_BYTES: raise SubmissionError('Submission exceeds 100MB')
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as z:
            infos=z.infolist()
            if len({i.filename for i in infos})!=len(infos):
                raise SubmissionError('Duplicate archive members')
            if sum(i.file_size for i in infos)>MAX_BYTES:
                raise SubmissionError('Expanded submission exceeds 100MB')
            if any(i.flag_bits&1 for i in infos): raise SubmissionError('Encrypted archive')
            if 'manifest.json' not in z.namelist(): raise SubmissionError('manifest.json required; old models not accepted')
            self.manifest=json.loads(z.read('manifest.json'))
            m=self.manifest
            if m.get('version')!=VERSION: raise SubmissionError('Old or unsupported version')
            if not isinstance(m.get('team'),str) or not m['team'].strip() or len(m['team'])>80:
                raise SubmissionError('Valid team name required')
            if not isinstance(m.get('sources'),list): raise SubmissionError('Source declarations required')
            if m.get('mode')=='hierarchical':
                members={'manifest.json','walk.onnx','flee.onnx'}
                shapes={'walk.onnx':(47,12),'flee.onnx':(19,3)}
            elif m.get('mode')=='direct':
                members={'manifest.json','direct.onnx'}
                shapes={'direct.onnx':(66,12)}
            else: raise SubmissionError('Mode must be hierarchical or direct')
            if set(z.namelist())!=members: raise SubmissionError('Unexpected archive members')
            self.policies={}
            for name,shape in shapes.items():
                data=z.read(name)
                if m.get('sha256',{}).get(name)!=hashlib.sha256(data).hexdigest():
                    raise SubmissionError('Model SHA256 mismatch')
                self.policies[name]=Policy(data,*shape)
        self.command=np.zeros(3,dtype=np.float32)

    def reset(self):
        self.command[:]=0

    def action(self,arena):
        if self.manifest['mode']=='direct':
            return self.policies['direct.onnx'](arena.direct_obs())
        if arena.ticks%10==0:
            self.command=self.policies['flee.onnx'](arena.high_obs())*np.array([.5,.3,1.])
        return self.policies['walk.onnx'](arena.walk_obs('flee',self.command))


def make_package(path,team,models,mode='hierarchical',sources=None):
    """Package exported float32 ONNX models, then re-open with the official validator."""
    path=Path(path)
    data={name:Path(file).read_bytes() for name,file in models.items()}
    manifest=dict(version=VERSION,team=team,mode=mode,sources=list(sources or []),
                  sha256={name:hashlib.sha256(b).hexdigest() for name,b in data.items()})
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        for name,b in data.items():z.writestr(name,b)
    Submission(path)
    return str(path)
