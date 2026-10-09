"""Isolated Colab dependencies; never replace packages in the notebook kernel."""
from pathlib import Path
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / '.venv-colab'
PYTHON = ENV / 'bin/python'


def worker_env():
    env = dict(os.environ, PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    return env


def prepare_runtime():
    if not PYTHON.is_file():
        venv.EnvBuilder(with_pip=True).create(ENV)
    def run(*args):
        subprocess.run([str(PYTHON), *args], cwd=ROOT, env=worker_env(), check=True)
    run('-u', 'scripts/colab_cpu_runtime.py')
    run('-m', 'pip', 'install', '-q', '-r', 'requirements.txt')
    flags = ['--ignore-requires-python'] if sys.version_info >= (3, 13) else []
    run('-m', 'pip', 'install', '-q', *flags, '-c', 'requirements-training.txt', 'mjswan==0.8.2')
    run('scripts/download_models.py')
    # Loading the real saved policy exercises Adam/Triton, not just imports.
    run('-c', 'from stable_baselines3 import PPO; '
        'PPO.load("models/pretrained/smooth_walk/walk_model.zip", device="cpu"); '
        'print("学習用環境のモデル読み込み確認: OK")')


if __name__ == '__main__':
    prepare_runtime()
