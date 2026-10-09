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


def run_logged(command):
    with subprocess.Popen(command, cwd=ROOT, env=worker_env(), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def prepare_runtime():
    if not PYTHON.is_file():
        venv.EnvBuilder(with_pip=False).create(ENV)
    # Colab does not ship ensurepip. Bootstrap only the venv via host pip.
    run_logged([sys.executable, "-m", "pip", "--python", str(PYTHON),
                "install", "-q", "pip"])
    def run(*args):
        run_logged([str(PYTHON), *args])
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
