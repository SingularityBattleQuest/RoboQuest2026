import json
import pytest
from scripts.learn_walk import fingerprints, require_qualified_walk


def test_qualification_is_bound_to_weights_normalization_and_settings(tmp_path):
    for name in ('walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json'):
        (tmp_path/name).write_bytes(b'original')
    report = {'passed': True, 'artifacts_sha256': fingerprints(tmp_path)}
    (tmp_path/'walk_qualification.json').write_text(json.dumps(report))
    assert require_qualified_walk(tmp_path) == report
    for name in report['artifacts_sha256']:
        (tmp_path/name).write_bytes(b'changed')
        with pytest.raises(RuntimeError, match='未合格'):
            require_qualified_walk(tmp_path)
        (tmp_path/name).write_bytes(b'original')
    report['passed'] = False
    (tmp_path/'walk_qualification.json').write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match='未合格'):
        require_qualified_walk(tmp_path)


def test_failed_training_does_not_publish_candidate(tmp_path, monkeypatch):
    from scripts import learn_walk as workflow
    from scripts import bootstrap_smooth_walk, tune_walk
    destination = tmp_path/'experiment'
    destination.mkdir()
    (destination/'walk_model.zip').write_bytes(b'previous good model')
    def train(folder, **kwargs):
        folder.mkdir()
        (folder/'walk_model.zip').write_bytes(b'failed candidate')
    monkeypatch.setattr(bootstrap_smooth_walk, 'train_smooth_walk', train)
    monkeypatch.setattr(tune_walk, 'evaluate', lambda *a, **kw: {'passed': False})
    monkeypatch.setattr(workflow, 'qualify_walk', lambda folder: {'passed': False})
    with pytest.raises(RuntimeError, match='合格基準を満たしません'):
        workflow.learn_walk(destination)
    assert (destination/'walk_model.zip').read_bytes() == b'previous good model'
    assert len(list(destination.glob('walk_training_*/walk_qualification.json'))) == 1
