"""JSON-only subprocess boundary for the teaching notebooks."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def execute(operation, args, kwargs):
    if operation == 'record_walk':
        import os
        if sys.platform == 'linux':
            os.environ.setdefault('MUJOCO_GL', 'egl')
        from scripts.record_walk import record
        record(*args, **kwargs)
        return str(args[1])
    from scripts import notebook_workflow as workflow
    from scripts import bootstrap_smooth_walk as smooth
    from roboquest.utils.reward_utils import WalkRewardConfig, FleeRewardConfig
    if operation == 'defaults':
        return dict(FLEE_PPO=workflow.FLEE_PPO, WALK_PPO=smooth.SMOOTH_PPO,
                    WALK_FORWARD_REWARD=smooth.SMOOTH_REWARD,
                    WALK_FORWARD_ENV=smooth.SMOOTH_ENV, WALK_FORWARD_STEPS=smooth.SMOOTH_STEPS,
                    WalkRewardConfig=asdict(WalkRewardConfig()), FleeRewardConfig=asdict(FleeRewardConfig()))
    if operation == 'train_policy':
        args[2] = (WalkRewardConfig if args[0] == 'walk' else FleeRewardConfig)(**args[2])
        return workflow.train_policy(*args, **kwargs)
    if operation in ('learn_walk', 'qualify_walk'):
        from scripts import learn_walk
        if kwargs.get('reward_config') is not None:
            kwargs['reward_config'] = WalkRewardConfig(**kwargs['reward_config'])
        return getattr(learn_walk, operation)(*args, **kwargs)
    if operation == 'train_smooth_walk':
        if kwargs.get('reward_config') is not None:
            kwargs['reward_config'] = WalkRewardConfig(**kwargs['reward_config'])
        return str(smooth.train_smooth_walk(*args, **kwargs))
    if operation in ('load_bundled_walk', 'evaluate_flee'):
        return getattr(workflow, operation)(*args, **kwargs)
    if operation == 'evaluate_walk':
        from scripts.tune_walk import evaluate
        return evaluate(*args, **kwargs)
    if operation in ('build_walk', 'build_flee'):
        from scripts import build_mjswan_viewer
        getattr(build_mjswan_viewer, operation)(*args, **kwargs)
        return None
    if operation == 'preview_saved_walk':
        from scripts.preview_saved_walk import preview_saved_walk
        path, has_stats = preview_saved_walk(*args, **kwargs)
        return [str(path), has_stats]
    raise ValueError(f'Unknown notebook operation: {operation}')


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    result = execute(request['operation'], request['args'], request['kwargs'])
    Path(sys.argv[2]).write_text(json.dumps(result), encoding='utf-8')
