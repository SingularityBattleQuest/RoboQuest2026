"""Execute the distributed notebook cells, including an export worker failure."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.notebook_workflow import load_bundled_walk


@pytest.fixture(params=['quickstart_ja', 'guide_and_experiments_ja'])
def cells(request):
    notebook = json.loads((ROOT / 'notebooks' / f'{request.param}.ipynb').read_text())
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']


def cell(cells, title):
    return next(s for s in cells if title in s.splitlines()[0])


def test_default_walk_trains_and_requires_qualification(cells, tmp_path):
    calls = []
    def learn(folder, **kwargs):
        calls.append((folder, kwargs))
        return {'passed': True}
    context = dict(Path=Path, SAVE_DIR=tmp_path, learn_walk=learn,
                   walk_timesteps=10000, seed=0, walk_cfg=None, walk_ppo={}, walk_num_envs=4)
    exec(cell(cells, '歩行モデルを学習する'), context)
    assert calls == [(tmp_path, dict(steps=10000, seed=0, reward_config=None, ppo_kwargs={}, num_envs=4))]
    assert context['walk_ready'] and not context['walk_export_ready']
    context['learn_walk'] = lambda *a, **kw: {'passed': False}
    with pytest.raises(RuntimeError, match='不合格'):
        exec(cell(cells, '歩行モデルを学習する'), context)
    assert not context['walk_ready']


def test_missing_bundle_never_starts_training(cells, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = dict(Path=Path, SAVE_DIR=tmp_path / 'saved', load_bundled_walk=load_bundled_walk)
    source = cell(cells, '歩行モデルを学習する').replace('retrain_walk = True', 'retrain_walk = False')
    with pytest.raises(FileNotFoundError, match='walk_model.zip'):
        exec(source, context)
    assert not context['SAVE_DIR'].exists()


def test_failed_export_worker_keeps_model_and_blocks_stale_viewer(cells, tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.learn_walk.require_qualified_walk', lambda folder: {'passed': True})
    source = tmp_path / 'walk_model.zip'
    source.write_bytes(b'saved model must survive')
    run = subprocess.run

    def failed_worker(args, **kwargs):
        # Actually terminate a subprocess; the notebook process must survive.
        return run([sys.executable, '-u', '-c',
                    "import os; print('export worker failed', flush=True); os._exit(7)"], **kwargs)

    monkeypatch.setattr(subprocess, 'run', failed_worker)
    context = dict(SAVE_DIR=tmp_path, walk_export_ready=True, TRAINING_PYTHON=sys.executable)
    with pytest.raises(RuntimeError, match='終了コード 7'):
        exec(cell(cells, 'ビューアー用に変換'), context)
    assert source.read_bytes() == b'saved model must survive'
    assert 'export worker failed' in (tmp_path / 'walk_viewer_export.log').read_text()
    for title in ['歩行ビューアー', '鬼ごっこビューアー']:
        with pytest.raises(RuntimeError, match='先に「ビューアー用に変換」'):
            exec(cell(cells, title), context)


def test_export_cell_runs_real_saved_model(cells, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('scripts.learn_walk.require_qualified_walk', lambda folder: {'passed': True})
    load_bundled_walk(tmp_path, ROOT / 'models/pretrained/smooth_walk')
    source = cell(cells, 'ビューアー用に変換').replace(
        "'/content/RoboQuest2026/webapp/models'", repr(str(tmp_path / 'web')))
    context = dict(SAVE_DIR=tmp_path, TRAINING_PYTHON=sys.executable)
    exec(source, context)
    assert context['walk_export_ready']
    assert (tmp_path / 'web/walk_policy_normalized.onnx').is_file()
    assert 'Verification max abs diff:' in (tmp_path / 'walk_viewer_export.log').read_text()


def test_all_code_cells_compile(cells):
    for source in cells:
        compile(source, '<notebook cell>', 'exec')
