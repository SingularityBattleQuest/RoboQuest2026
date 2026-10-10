"""Notebook actions for the approved physical one-on-one competition."""
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from competition.arena import Arena,gait_action
from competition.rules import VERSION
from competition.submission import Policy,make_package,Submission
from scripts.export_for_web import export_normalized_policy_onnx,NormalizedPolicyONNX
from scripts.bootstrap_smooth_walk import SMOOTH_ENV,SMOOTH_REWARD,SMOOTH_PPO
from roboquest.envs.go2_walk_env import Go2WalkEnv
from roboquest.utils.reward_utils import WalkRewardConfig


class CompetitionWalk(gym.ObservationWrapper):
    """Official 45 physical features plus a public 1.5Hz phase clock."""
    def __init__(self,**kwargs):
        super().__init__(Go2WalkEnv(**kwargs))
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(47,),dtype=np.float32)

    def observation(self,obs):
        phase=2*math.pi*1.5*self.unwrapped.data.time
        return np.concatenate([obs,[math.sin(phase),math.cos(phase)]]).astype(np.float32)

    def set_vel_cmd(self,*args):
        self.unwrapped.set_vel_cmd(*args)

    def _get_obs(self):
        return self.observation(self.unwrapped._get_obs())

    @property
    def robot_xy(self):return self.unwrapped.robot_xy

    @property
    def data(self):return self.unwrapped.data


def _fit(model,env,observations,targets,updates,seed):
    rng=np.random.default_rng(seed)
    env.obs_rms.update(np.asarray(observations))
    x=torch.as_tensor(env.normalize_obs(np.asarray(observations,dtype=np.float32)))
    y=torch.as_tensor(np.asarray(targets,dtype=np.float32))
    params=list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())
    optimizer=torch.optim.Adam(params,lr=3e-4)
    for update in range(updates):
        idx=torch.as_tensor(rng.integers(0,len(x),512))
        loss=(model.policy._predict(x[idx],deterministic=True)-y[idx]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (update+1)%1000==0:print(f'見本学習 {update+1}/{updates}: MSE {loss.item():.6f}',flush=True)
    with torch.no_grad():model.policy.log_std.fill_(-3.)
    model.policy.optimizer.state.clear()
    env.training=False


def _verified_export(model,env,directory,name,observations):
    model.save(directory/f'{name}_model')
    env.save(str(directory/f'{name}_model_vecnorm.pkl'))
    path=directory/f'{name}.onnx'
    export_normalized_policy_onnx(directory/f'{name}_model',directory/f'{name}_model_vecnorm.pkl',path)
    p=Policy(path.read_bytes(),model.observation_space.shape[0],model.action_space.shape[0])
    wrapper=NormalizedPolicyONNX(model.policy,env.obs_rms,env.clip_obs,env.epsilon)
    error=0.
    for obs in observations[:100]:
        with torch.no_grad():expected=wrapper(torch.as_tensor(np.asarray(obs,dtype=np.float32)[None])).numpy()[0]
        error=max(error,float(np.max(np.abs(expected-p(obs)))))
    if error>1e-4:raise RuntimeError(f'正規化込みONNX出力の不一致: {error}')
    print(f'正規化込みONNX照合: 最大誤差 {error:.8g}',flush=True)
    return p,error


def prepare_oni(save_dir):
    """Verify physical chasing from all four cardinal starts, not a canned animation."""
    folder=Path(save_dir);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    arena=Arena()
    try:
        for spawn in ('east','north','west','south'):
            arena.reset(90,spawn)
            start=arena.position('oni').copy();minheight=1.
            while not arena.reason:
                arena.step(gait_action(arena.ticks*.02,0.))
                minheight=min(minheight,float(arena.position('oni')[2]))
            row=arena.result()
            row.update(oni_distance_m=float(np.linalg.norm(arena.position('oni')[:2]-start[:2])),oni_min_height=minheight)
            rows.append(row)
            print(f"四脚の鬼 {spawn}: {row['survived_seconds']:.2f}s、{row['reason']}、最低胴体高 {minheight:.3f}m",flush=True)
    finally:arena.close()
    passed=all(r['reason']=='captured' and r['oni_min_height']>=.15 and r['oni_distance_m']>1 for r in rows)
    report=dict(version=VERSION,passed=passed,controller='fixed feedback + inverse-kinematic trot',rows=rows)
    (folder/'oni_verification.json').write_text(json.dumps(report,indent=2))
    if not passed:raise RuntimeError('鬼の追跡試験に不合格。次へ進めません。')
    return report


def train_competition_walk(save_dir,seed=0,sample_steps=30000,updates=4000,ppo_steps=10000):
    folder=Path(save_dir)/'competition_walk';folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    settings={**SMOOTH_ENV,'command_mode':'uniform',
              'command_ranges':{'vx':[-.4,.45],'vy':[-.15,.15],'omega':[-.7,.7]}}
    reward=WalkRewardConfig(**SMOOTH_REWARD)
    env=VecNormalize(make_vec_env(lambda:CompetitionWalk(reward_config=reward,**settings),n_envs=4,seed=seed))
    model=PPO('MlpPolicy',env,seed=seed,**SMOOTH_PPO)
    teacher=CompetitionWalk(randomize_cmd=False,**settings)
    observations=[];targets=[]
    print('本番用の歩行を未学習モデルから学習します（前後進・旋回・横移動・停止）。',flush=True)
    try:
        for step in range(sample_steps):
            local=step%250
            if local==0:
                choice=step//250%8
                command=[(.4,0.,0.),(0.,0.,.6),(0.,0.,-.6),(-.3,0.,0.),
                         (.3,0.,.4),(.3,0.,-.4),(0.,.12,0.),(0.,0.,0.)][choice]
                teacher.set_vel_cmd(*command)
                obs,_=teacher.reset(seed=1000+seed+step)
            action=gait_action(local*.02,command[0],command[2],command[1])
            observations.append(obs.copy());targets.append(action)
            obs,_,fell,_,_=teacher.step(action)
            if fell:raise RuntimeError(f'本番歩行の見本が転倒しました: {command}')
            if step and step%10000==0:print(f'歩行見本 {step}/{sample_steps}',flush=True)
    finally:teacher.close()
    try:
        _fit(model,env,observations,targets,updates,seed)
        # Teach restarting from poses reached by the student's own controller.
        # Teacher-only trajectories otherwise permit a stationary fixed point.
        for round_index in range(3):
            extra_obs=[];extra_targets=[]
            probe=CompetitionWalk(randomize_cmd=False,**settings)
            try:
                for command in [( .4,0.,0.),(-.3,0.,0.),(0.,0.,.6),(0.,0.,-.6),(0.,.12,0.),(.3,0.,.4),(.3,0.,-.4)]:
                    probe.set_vel_cmd(0.,0.,0.)
                    obs,_=probe.reset(seed=2000+seed+round_index)
                    for cmd,count in [((0.,0.,0.),500),(command,500),((0.,0.,0.),500),(command,500)]:
                        probe.set_vel_cmd(*cmd);obs=probe._get_obs().astype(np.float32)
                        for step in range(count):
                            target=gait_action(probe.data.time,cmd[0],cmd[2],cmd[1])
                            extra_obs.append(obs.copy());extra_targets.append(target)
                            action=model.predict(env.normalize_obs(obs),deterministic=True)[0] if cmd==(0.,0.,0.) else target
                            obs,_,fell,_,_=probe.step(action)
                            if fell:break
            finally:probe.close()
            # Keep the normalization fixed after collecting the original data.
            x=torch.as_tensor(env.normalize_obs(np.asarray(observations,dtype=np.float32)))
            y=torch.as_tensor(np.asarray(targets,dtype=np.float32))
            ex=torch.as_tensor(env.normalize_obs(np.asarray(extra_obs,dtype=np.float32)))
            ey=torch.as_tensor(np.asarray(extra_targets,dtype=np.float32))
            actor=list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())
            optimizer=torch.optim.Adam(actor,lr=1e-4)
            for update in range(2000):
                old=torch.as_tensor(rng.integers(0,len(x),256));new=torch.as_tensor(rng.integers(0,len(ex),256))
                loss=(model.policy._predict(torch.cat([x[old],ex[new]]),deterministic=True)-torch.cat([y[old],ey[new]])).square().mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
            print(f'本番用の再発進学習 {round_index+1}/3 完了',flush=True)
        print(f'歩行PPO {ppo_steps}ステップ',flush=True)
        model.learn(total_timesteps=ppo_steps,progress_bar=False)
        (folder/'walk_params.json').write_text(json.dumps({'version':VERSION,'walk_env_kwargs':settings}))
        policy,error=_verified_export(model,env,folder,'walk',observations)
    finally:env.close()
    rows=[]
    for command in [(0.,0.,0.),(.4,0.,0.),(-.3,0.,0.),(0.,0.,.5),(0.,0.,-.5),(0.,.1,0.)]:
        probe=CompetitionWalk(randomize_cmd=False,**settings)
        try:
            probe.set_vel_cmd(*command);obs,_=probe.reset(seed=500)
            start=probe.robot_xy.copy();yaw=0.;fell=False
            for step in range(1000):
                obs,_,fell,_,_=probe.step(policy(obs));yaw+=probe.data.qvel[5]*.02
                if fell:break
            distance=float(np.linalg.norm(probe.robot_xy-start))
            passed=not fell
            if command[0]:passed=passed and distance>=2.
            if command[2]:passed=passed and yaw*command[2]>1.
            rows.append(dict(command=command,passed=bool(passed),fell=bool(fell),distance_m=distance,yaw_radians=float(yaw)))
            print('歩行検証',rows[-1],flush=True)
        finally:probe.close()
    report=dict(version=VERSION,passed=all(r['passed'] for r in rows),rows=rows,export_error=error)
    (folder/'qualification.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('本番歩行の検証に不合格。学習回数・見本を修正してください。')
    return report


def escape_example(obs):
    """Training-only potential-field guidance. Not used by scoring inference."""
    heading=math.atan2(obs[6],obs[5]);c,s=math.cos(heading),math.sin(heading)
    rel=np.array([c*obs[0]-s*obs[1],s*obs[0]+c*obs[1]])
    away=-rel/max(float(np.linalg.norm(rel)),.1)
    walls=np.maximum(obs[10:14],.2)
    force=away+np.array([1/walls[2]**2-1/walls[0]**2,1/walls[3]**2-1/walls[1]**2])*.5
    force+=.3*np.array([-away[1],away[0]])
    error=math.atan2(math.sin(math.atan2(force[1],force[0])-heading),math.cos(math.atan2(force[1],force[0])-heading))
    return np.array([.8*max(0.,math.cos(error)),0.,np.clip(1.5*error,-.7,.7)],np.float32)


def train_competition_flee(save_dir,seed=0,sample_steps=12000,updates=2000,ppo_steps=10000,reward=None):
    from competition.gym_env import FleeEnv
    folder=Path(save_dir);walk_folder=folder/'competition_walk'
    q=json.loads((walk_folder/'qualification.json').read_text())
    if not q['passed'] or q['version']!=VERSION:raise RuntimeError('本番用歩行を先に合格させてください。')
    walk=Policy((walk_folder/'walk.onnx').read_bytes(),47,12)
    torch.set_num_threads(1);torch.manual_seed(seed)
    env=VecNormalize(make_vec_env(lambda:FleeEnv(walk,reward),n_envs=2,seed=seed))
    settings=dict(learning_rate=1e-5,n_steps=512,batch_size=128,n_epochs=5,
                  ent_coef=0.,policy_kwargs={'net_arch':[128,128]},device='cpu')
    model=PPO('MlpPolicy',env,seed=seed,**settings)
    teacher=FleeEnv(walk,reward)
    observations=[];targets=[]
    obs,_=teacher.reset(seed=seed)
    print('実際の四脚の鬼から逃げる見本を収集します。',flush=True)
    try:
        for step in range(sample_steps):
            action=escape_example(obs)
            observations.append(obs.copy());targets.append(action)
            obs,_,terminated,truncated,_=teacher.step(action)
            if terminated or truncated:obs,_=teacher.reset()
            if step and step%3000==0:print(f'逃げ見本 {step}/{sample_steps}',flush=True)
    finally:teacher.close()
    try:
        _fit(model,env,observations,targets,updates,seed)
        print(f'逃げPPO {ppo_steps}ステップ',flush=True)
        model.learn(total_timesteps=ppo_steps,progress_bar=False)
        policy,error=_verified_export(model,env,folder,'flee',observations)
        (folder/'competition_training.json').write_text(json.dumps(dict(version=VERSION,seed=seed,
            sample_steps=sample_steps,updates=updates,ppo_steps=int(model.num_timesteps),export_error=error),indent=2))
    finally:env.close()
    return dict(version=VERSION,trained=True,export_error=error)


def package_competition(save_dir,team,sources=None):
    folder=Path(save_dir)
    return make_package(folder/'submission.zip',team,
        {'walk.onnx':folder/'competition_walk/walk.onnx','flee.onnx':folder/'flee.onnx'},sources=sources)


def evaluate_competition(save_dir,video=False):
    from competition.run_match import isolated_evaluate
    folder=Path(save_dir)
    report=isolated_evaluate(folder/'submission.zip',video=folder/'competition_match.mp4' if video else None)
    (folder/'competition_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def record_oni_demo(save_dir):
    import imageio.v2 as imageio
    folder=Path(save_dir);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'oni_demo.mp4'
    arena=Arena();arena.reset(90,'east')
    try:
        with imageio.get_writer(path,fps=10) as writer:
            writer.append_data(arena.render())
            while not arena.reason:
                arena.step(gait_action(arena.ticks*.02,0.))
                if arena.ticks%5==0:writer.append_data(arena.render())
    finally:arena.close()
    return str(path)
