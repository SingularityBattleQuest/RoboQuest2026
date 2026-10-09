from importlib.metadata import PackageNotFoundError
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import colab_cpu_runtime as runtime


def test_replaces_cuda_torch_and_removes_leftover_triton(monkeypatch):
    installed = {'torch': '2.14.0', 'triton': '3.8.0'}
    calls = []
    monkeypatch.setattr(runtime.sys, 'platform', 'linux')
    monkeypatch.setattr(runtime.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(runtime, 'version', installed.__getitem__)

    def run(args, check):
        assert check
        calls.append(args)
        if 'install' in args:
            installed['torch'] = '2.14.0+cpu'

    monkeypatch.setattr(runtime.subprocess, 'run', run)
    runtime.prepare_cpu_torch()
    assert calls[0][-1] == 'torch==2.14.0+cpu'
    assert 'https://download.pytorch.org/whl/cpu' in calls[0]
    assert calls[1][-3:] == ['uninstall', '-y', 'triton']


def test_cpu_setup_is_idempotent(monkeypatch):
    monkeypatch.setattr(runtime.sys, 'platform', 'linux')
    monkeypatch.setattr(runtime.platform, 'machine', lambda: 'x86_64')

    def version(name):
        if name == 'torch':
            return '2.14.0+cpu'
        raise PackageNotFoundError(name)

    monkeypatch.setattr(runtime, 'version', version)
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('An already prepared runtime must not reinstall packages')))
    runtime.prepare_cpu_torch()
