"""Standard-library-only notebook API; native packages run in an isolated process."""
from dataclasses import asdict, field, is_dataclass, make_dataclass
from functools import partial
from pathlib import Path
import json
import subprocess
import tempfile

from scripts.colab_runtime import PYTHON, ROOT, worker_env


def _encode(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f'Unsupported notebook parameter: {type(value).__name__}')


def call(operation, *args, **kwargs):
    with tempfile.TemporaryDirectory(prefix='rq_call_') as directory:
        request, response = Path(directory) / 'request.json', Path(directory) / 'response.json'
        request.write_text(json.dumps(dict(operation=operation, args=args, kwargs=kwargs), default=_encode))
        process = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'scripts/colab_worker.py'),
                                    str(request), str(response)], cwd=ROOT, env=worker_env(),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
                process.wait()
            raise
        finally:
            process.stdout.close()
        if code:
            raise RuntimeError(f'{operation} が失敗しました（終了コード {code}）。上のログを確認してください。')
        return json.loads(response.read_text())


def notebook_api():
    defaults = call('defaults')
    # Keep editable dataclass settings in the kernel without importing MuJoCo,
    # NumPy or Torch through roboquest's package initializer.
    for name in ('WalkRewardConfig', 'FleeRewardConfig'):
        defaults[name] = make_dataclass(name, [(key, type(value), field(default=value))
                                              for key, value in defaults[name].items()])
    for name in ('load_bundled_walk', 'train_policy', 'train_smooth_walk', 'evaluate_flee', 'evaluate_walk'):
        defaults[name] = partial(call, name)
    return defaults


def build_walk(*args, **kwargs):
    return call('build_walk', *args, **kwargs)


def build_flee(*args, **kwargs):
    return call('build_flee', *args, **kwargs)


def preview_saved_walk(*args, **kwargs):
    path, has_stats = call('preview_saved_walk', *args, **kwargs)
    return Path(path), has_stats
