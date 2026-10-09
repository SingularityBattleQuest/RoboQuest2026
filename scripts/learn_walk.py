"""Train from random weights, then certify the introductory forward/stop task."""
import hashlib
import json
from pathlib import Path
import shutil
import time


def fingerprints(folder):
    return {name: hashlib.sha256((Path(folder)/name).read_bytes()).hexdigest()
            for name in ('walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json')}


def qualify_walk(folder):
    from scripts.tune_walk import evaluate
    from scripts.evaluate_walk_transitions import evaluate_transitions
    folder = Path(folder)
    # These holdouts are not used to fit or select model weights.
    fixed = evaluate(folder, seeds=(100, 101, 102), seconds=20,
                     commands={'stand': (0., 0., 0.), 'forward': (.4, 0., 0.)})
    transitions = evaluate_transitions(folder, seeds=(200, 203, 209), seconds=20, verbose=False)
    gait = all(row['distance_m'] >= 5.0
               and row['dominant_joint_frequency_hz'] is not None
               and .5 <= row['dominant_joint_frequency_hz'] <= 3.0
               and row['joint_speed_rms_rad_s'] >= .2
               for row in fixed['rows'] if row['command'] == 'forward')
    passed = bool(fixed['passed'] and fixed['posture_passed'] and fixed['startup_passed']
                  and transitions['passed'] and gait)
    report = dict(passed=passed, task='forward_0.4mps_stop_restart',
                  unsupported_commands=['backward', 'lateral', 'turn'],
                  fixed=fixed, transitions=transitions, gait_passed=gait,
                  criteria=dict(forward_distance_min_m=5., duration_seconds=20,
                                joint_frequency_hz=[.5, 3.], joint_speed_rms_min=.2),
                  artifacts_sha256=fingerprints(folder))
    (folder/'walk_qualification.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def learn_walk(save_dir, steps=10000, seed=0, reward_config=None, ppo_kwargs=None, num_envs=4):
    """Publish a model only after real rollout checks pass; keep failed runs."""
    from scripts.bootstrap_smooth_walk import train_smooth_walk
    from scripts.tune_walk import evaluate
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run = destination / f'walk_training_{time.time_ns()}'
    print('未学習のモデルから歩行を学習します。同梱モデルは使いません。', flush=True)
    train_smooth_walk(run, steps=steps, seed=seed, reward_config=reward_config,
                      ppo_kwargs=ppo_kwargs, num_envs=num_envs)
    print('学習前と学習後を、未使用の初期姿勢で評価します。', flush=True)
    before = evaluate(run/'untrained', seeds=(100, 101, 102), seconds=20,
                      commands={'stand': (0., 0., 0.), 'forward': (.4, 0., 0.)})
    report = qualify_walk(run)
    report['untrained'] = before
    report['training_run'] = str(run)
    (run/'walk_qualification.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']:
        raise RuntimeError(f'歩行の合格基準を満たしませんでした。学習済みとして採用しません。'
                           f'詳細: {run / "walk_qualification.json"}。初期設定を確認してください。')
    for name in ('walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json',
                 'walk_smooth_curriculum.json', 'walk_qualification.json',
                 'walk_transition_evaluation.json'):
        shutil.copy2(run/name, destination/name)
    forwards = [r for r in report['fixed']['rows'] if r['command']=='forward']
    print(f"✅ 前進・停止・再発進に合格。平均前進速度 "
          f"{sum(r['mean_velocity'][0] for r in forwards)/len(forwards):.3f} m/s、"
          f"20秒の移動距離 {min(r['distance_m'] for r in forwards):.2f} m以上、転倒0回。", flush=True)
    print('これは前進歩行の入門課題です。後退・横移動・旋回の獲得は含みません。', flush=True)
    return report


def require_qualified_walk(folder):
    """A stale PASS from another checkpoint must never authorize a new model."""
    folder = Path(folder)
    report = json.loads((folder/'walk_qualification.json').read_text())
    actual = fingerprints(folder)
    if not report['passed'] or report['artifacts_sha256'] != actual:
        raise RuntimeError('この歩行モデルは未合格です。歩行学習・評価を完了してください。')
    return report
