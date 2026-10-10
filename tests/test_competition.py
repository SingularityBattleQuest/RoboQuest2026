"""Physical competition, model validation, and tie-break regression coverage."""
import json
import zipfile
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
import onnx
from onnx import helper, TensorProto
import pytest
from competition.arena import Arena, gait_action
from competition.rules import Rules
from competition.submission import Policy, Submission, SubmissionError, make_package
from competition.run_match import tournament


def constant_model(inputs, outputs, value=0.):
    graph=helper.make_graph([helper.make_node('Constant',[],['action'],value=helper.make_tensor('v',TensorProto.FLOAT,[1,outputs],[value]*outputs))],
        'test',[helper.make_tensor_value_info('obs',TensorProto.FLOAT,[1,inputs])],
        [helper.make_tensor_value_info('action',TensorProto.FLOAT,[1,outputs])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)])
    model.ir_version=10
    return model.SerializeToString()


def test_fixed_rules_reject_training_overrides():
    for kwargs in [dict(duration=5),dict(oni_speed=1),dict(fall_height=.1),dict(spawn_distance=1)]:
        with pytest.raises(ValueError):Rules(**kwargs)


def test_two_articulated_robots_and_observation_frames():
    a=Arena()
    try:
        assert a.model.nu==24 and a.model.nq==38
        a.reset(100,'north')
        np.testing.assert_allclose(a.position('oni')[:2],[0,2],atol=1e-12)
        assert a.high_obs().shape==(19,) and a.walk_obs('flee',[0,0,0]).shape==(47,)
        assert a.direct_obs().shape==(66,)
        np.testing.assert_allclose(a.high_obs()[:3],[0,2,2],atol=1e-6)
        q=a.parts['flee']['q']
        a.data.qpos[q+3:q+7]=[np.sqrt(.5),0,0,np.sqrt(.5)]
        mujoco.mj_forward(a.model,a.data)
        np.testing.assert_allclose(a.high_obs()[:2],[2,0],atol=1e-6)
        np.testing.assert_allclose(a.walk_obs('flee',[0,0,0])[-2:],[0,1])
    finally:a.close()


def test_repeatable_real_pursuit_and_contact_capture():
    a=Arena()
    try:
        rows=[]
        for repeat in range(2):
            a.reset(100,'east')
            while not a.reason:a.step(gait_action(a.data.time,0.))
            assert a.reason=='captured' and 2<a.data.time<15
            assert a.position('oni')[2]>.15
            assert abs(a.position('oni')[0]-2)>1
            assert any({a._geom_role.get(int(c.geom1)),a._geom_role.get(int(c.geom2))}=={'oni','flee'} and c.efc_address>=0 for c in a.data.contact[:a.data.ncon])
            rows.append((a.result(),a.data.qpos.copy()))
        assert rows[0][0]==rows[1][0]
        np.testing.assert_array_equal(rows[0][1],rows[1][1])
    finally:a.close()


def test_field_exit_and_oni_fall_does_not_stop_clock():
    a=Arena()
    try:
        a.reset()
        q=a.parts['oni']['q'];a.data.qpos[q+2]=.1
        mujoco.mj_forward(a.model,a.data)
        a.step(gait_action(0,0))
        assert a.data.time>0 and a.reason is None
        q=a.parts['flee']['q'];a.data.qpos[q]=2.8
        mujoco.mj_forward(a.model,a.data)
        assert a.step(gait_action(.02,0))['reason']=='out_of_bounds'
    finally:a.close()


def test_old_and_nonfinite_models_rejected(tmp_path):
    old=tmp_path/'old.zip'
    with zipfile.ZipFile(old,'w') as z:z.writestr('policy.pth',b'old')
    with pytest.raises(SubmissionError,match='old models'):Submission(old)
    with pytest.raises(SubmissionError):Policy(constant_model(48,12),66,12)
    with pytest.raises(SubmissionError,match='NaN'):Policy(constant_model(66,12,float('nan')),66,12)
    with pytest.raises(SubmissionError,match='within'):Policy(constant_model(66,12,2),66,12)


def test_valid_direct_package_and_tampered_hash(tmp_path):
    model=tmp_path/'direct.onnx';model.write_bytes(constant_model(66,12))
    package=tmp_path/'team.zip'
    make_package(package,'test',{'direct.onnx':model},mode='direct')
    a=Arena()
    try:np.testing.assert_array_equal(Submission(package).action(a),np.zeros(12))
    finally:a.close()
    with zipfile.ZipFile(package) as z:manifest=json.loads(z.read('manifest.json'))
    manifest['sha256']['direct.onnx']='bad'
    with zipfile.ZipFile(package,'w') as z:
        z.writestr('manifest.json',json.dumps(manifest));z.writestr('direct.onnx',model.read_bytes())
    with pytest.raises(SubmissionError,match='SHA256'):Submission(package)


def test_ties_use_same_submission_and_shared_rank():
    calls=[]
    def evaluate(path,speed=.3):
        calls.append((path,speed))
        return dict(team=path,score_ticks={'a':10,'b':10,'c':5}[path])
    with patch('competition.run_match.isolated_evaluate',evaluate):result=tournament(['a','b','c'])
    assert [(r['team'],r['rank']) for r in result]==[('a',1),('b',1),('c',3)]
    assert calls==[('a',.3),('b',.3),('c',.3),('a',.4),('b',.4),('a',.5),('b',.5)]


def test_full_sixty_seconds_and_fall_duration_boundary():
    a=Arena()
    try:
        a.oni_action=lambda:gait_action(a.data.time,0.)
        a.reset(103,'west')
        for _ in range(3000):
            result=a.step(gait_action(a.data.time,0.))
            if a.reason:break
        assert result['reason']=='survived'
        assert result['survived_seconds']==pytest.approx(60.)
        assert a.ticks==3000
        with pytest.raises(RuntimeError):a.step(np.zeros(12))
        a.reset()
        a.fall_ticks=24
        q=a.parts['flee']['q'];a.data.qpos[q+2]=.04
        mujoco.mj_forward(a.model,a.data)
        assert a.step(np.zeros(12))['reason']=='fell'
    finally:a.close()
