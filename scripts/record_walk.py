"""Record the saved neural policy in MuJoCo, using its saved normalization/settings."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from roboquest.envs.go2_walk_env import Go2WalkEnv, CONTROL_DT


def record(folder, output, command=(.4, 0, 0), seconds=12, seed=100, fps=25, width=640, height=480):
    torch.set_num_threads(1)
    folder, output = Path(folder), Path(output)
    params = json.loads((folder/'walk_params.json').read_text())
    env_kwargs = params.get('walk_env_kwargs', {})
    raw = Go2WalkEnv(randomize_cmd=False, **env_kwargs)
    base = make_vec_env(lambda: Go2WalkEnv(randomize_cmd=False, **env_kwargs), n_envs=1)
    norm = VecNormalize.load(str(folder/'walk_model_vecnorm.pkl'), base)
    norm.training = False
    norm.norm_reward = False
    model = PPO.load(folder/'walk_model', device='cpu')
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.5, 125., -18.
    raw.set_vel_cmd(*command)
    obs, _ = raw.reset(seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, round(1 / (fps * CONTROL_DT)))
    try:
        raw._renderer = mujoco.Renderer(raw.model, height=height, width=width)
        print(f'歩行動画を作成中（{seconds}秒、{width}×{height}）...', flush=True)
        with imageio.get_writer(output, fps=1/(stride*CONTROL_DT), codec='libx264') as writer:
            for step in range(round(seconds/CONTROL_DT)):
                action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
                obs, _, terminated, _, _ = raw.step(action)
                if step % stride == 0:
                    camera.lookat[:] = raw.data.qpos[:3]
                    raw._renderer.update_scene(raw.data, camera=camera)
                    # Software EGL on Colab CPU spends most time on mesh
                    # shadows/reflections; they do not affect the simulation.
                    raw._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    raw._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                    writer.append_data(raw._renderer.render())
                if step and step % 100 == 0:
                    print(f'  動画 {step*CONTROL_DT:.0f}/{seconds} 秒', flush=True)
                if terminated:
                    break
    finally:
        raw.close()
        norm.close()
    print(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--output', default='artifacts/walk_forward.mp4')
    args = parser.parse_args()
    record(args.folder, args.output)
