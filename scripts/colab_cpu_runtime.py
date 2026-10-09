"""Select the CPU PyTorch build for the CPU-based Colab teaching workflow.

On Colab Python 3.13, the CUDA build's Triton import can segfault while
constructing SB3's Adam optimizer, even when PPO uses device='cpu'.
Do not import torch in this installer: replacing a loaded native module requires
a runtime restart. The notebook calls this before importing the training stack.
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys


def prepare_cpu_torch():
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        print('Colab用CPU版の選択はLinux x86_64のみで行います。', flush=True)
        return
    constraints = Path(__file__).resolve().parents[1] / 'requirements-training.txt'
    pinned = next(line.split('==', 1)[1].strip() for line in
                  constraints.read_text().splitlines() if line.startswith('torch=='))
    wanted = pinned + '+cpu'
    try:
        installed = version('torch')
    except PackageNotFoundError:
        installed = None
    if installed != wanted:
        print(f'PyTorch公式CPU版 {wanted} をインストールします。', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--index-url', 'https://download.pytorch.org/whl/cpu',
                        f'torch=={wanted}'], check=True)
    # CPU wheels do not depend on Triton, but Colab or an earlier setup may
    # leave it installed. Torch discovers it even when CUDA is not used.
    try:
        version('triton')
    except PackageNotFoundError:
        pass
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'triton'],
                       check=True)
    print(f'CPU学習用PyTorch: {version("torch")}', flush=True)


if __name__ == '__main__':
    prepare_cpu_torch()
