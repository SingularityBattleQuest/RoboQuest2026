"""Exercise the real subprocess boundary while the parent has incompatible imports."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_training_and_evaluation_do_not_import_parent_native_packages(tmp_path):
    # Fake already-loaded packages deliberately cannot be used for training.
    # The actual worker uses this test interpreter's installed numerical stack.
    source = '''
import sys, types, json
from pathlib import Path
for name in ('numpy', 'torch', 'mujoco', 'pandas', 'matplotlib'):
    module = types.ModuleType(name)
    module.__version__ = 'incompatible-parent-version'
    sys.modules[name] = module
from scripts import colab_client as client
client.PYTHON = Path(sys.executable)
api = client.notebook_api()
folder = Path(sys.argv[1])
api['load_bundled_walk'](folder, 'models/pretrained/smooth_walk')
cfg = api['FleeRewardConfig'](survival_weight=0.75)
settings = dict(api['FLEE_PPO'], n_steps=16, batch_size=16, n_epochs=1)
api['train_policy']('flee', folder, cfg, 32, 1, settings, progress_bar=False)
results = api['evaluate_flee'](folder, seeds=[100])
assert len(results) == 1 and results[0]['survived_seconds'] > 0.2
params = json.loads((folder/'flee_params.json').read_text())
assert params['reward_config']['survival_weight'] == 0.75
for name in ('numpy', 'torch', 'mujoco', 'pandas', 'matplotlib'):
    assert sys.modules[name].__version__ == 'incompatible-parent-version'
try:
    api['evaluate_flee'](folder/'missing')
except RuntimeError as error:
    assert 'evaluate_flee' in str(error)
else:
    raise AssertionError('worker failure must reach notebook')
print('ISOLATED_TRAIN_EVALUATE_PASS')
'''
    result = subprocess.run([sys.executable, '-c', source, str(tmp_path)], cwd=ROOT,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ISOLATED_TRAIN_EVALUATE_PASS' in result.stdout


def test_notebooks_use_isolated_setup_and_viewers():
    for name in ('quickstart_ja', 'guide_and_experiments_ja'):
        notebook = json.loads((ROOT/'notebooks'/f'{name}.ipynb').read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        assert 'prepare_runtime()' in code
        assert 'pip' not in code
        assert 'from scripts.colab_client import notebook_api' in code
        assert 'import mjswan' not in code
        assert 'from scripts.build_mjswan_viewer import' not in code
        assert 'from scripts.preview_saved_walk import' not in code
        assert 'str(TRAINING_PYTHON)' in code
        assert '_changed_loaded' not in code


def test_setup_bootstraps_only_the_venv(monkeypatch, tmp_path):
    from scripts import colab_runtime as runtime
    target = tmp_path / 'venv'
    monkeypatch.setattr(runtime, 'ENV', target)
    monkeypatch.setattr(runtime, 'PYTHON', target/'bin/python')
    created, commands = [], []
    class Builder:
        def __init__(self, **kwargs):
            assert kwargs == {'with_pip': False}
        def create(self, path):
            created.append(path)
    monkeypatch.setattr(runtime.venv, 'EnvBuilder', Builder)
    monkeypatch.setattr(runtime, 'run_logged', commands.append)
    runtime.prepare_runtime()
    assert created == [target]
    assert commands[0][:5] == [sys.executable, '-m', 'pip', '--python', str(target/'bin/python')]
    assert all(command[0] == str(target/'bin/python') for command in commands[1:])
    assert all('--system-site-packages' not in command for command in commands)


def test_worker_environment_does_not_inherit_package_paths(monkeypatch):
    from scripts.colab_runtime import worker_env
    monkeypatch.setenv('PYTHONPATH', '/host-packages')
    monkeypatch.setenv('PYTHONHOME', '/host-python')
    env = worker_env()
    assert 'PYTHONPATH' not in env and 'PYTHONHOME' not in env
    assert env['PYTHONNOUSERSITE'] == '1'
