"""Official one-on-one evaluation. python -m competition.run_match --submission team.zip"""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def evaluate(submission_path,oni_speed=.3,video=None):
    from competition.arena import Arena
    from competition.rules import Rules,SEEDS,SPAWNS
    from competition.scene import scene_digest
    from competition.submission import Submission,SubmissionError
    package=Submission(submission_path)
    arena=Arena(Rules(oni_speed=oni_speed))
    results=[]
    writer=None
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        writer=imageio.get_writer(str(video),fps=10)
    try:
        for seed in SEEDS:
            for spawn in SPAWNS:
                arena.reset(seed,spawn);package.reset()
                if writer and seed==SEEDS[0] and spawn==SPAWNS[0]:
                    writer.append_data(arena.render())
                while not arena.reason:
                    try:action=package.action(arena)
                    except SubmissionError as e:
                        arena.reason='model_error'
                        result=arena.result();result['error']=str(e)
                        break
                    result=arena.step(action)
                    if result['reason']=='physics_error':
                        raise RuntimeError('Infrastructure physics error: match must be replayed')
                    if writer and seed==SEEDS[0] and spawn==SPAWNS[0] and arena.ticks%5==0:
                        writer.append_data(arena.render())
                results.append(result)
                if writer and seed==SEEDS[0] and spawn==SPAWNS[0] and arena.ticks%5:
                    writer.append_data(arena.render())
                print(f"{seed}/{spawn}: {result['survived_seconds']:.3f}s {result['reason']}",flush=True)
    finally:
        arena.close()
        if writer:writer.close()
    score_ticks=sum(round(r['survived_seconds']/.002) for r in results)
    return dict(team=package.manifest['team'],submission_sha256=package.sha256,
                oni_speed=oni_speed,oni_sha256=arena.oni_digest,episodes=results,score_ticks=score_ticks,
                mean_survival_seconds=score_ticks*.002/len(results),
                completions=sum(r['reason']=='survived' for r in results),
                scene_sha256=scene_digest(),
                code_sha256=hashlib.sha256(b''.join(
                    p.name.encode()+p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest(),
                rule_sha256=arena.rules.digest,rule_version=arena.rules.version,
                environment={'python':platform.python_version(),'platform':platform.platform(),
                             **{p:version(p) for p in ('mujoco','numpy','onnx','onnxruntime')}})


def isolated_evaluate(path,oni_speed=.3,video=None):
    with tempfile.TemporaryDirectory(prefix='rq_match_') as tmp:
        output=Path(tmp)/'result.json'
        command=[sys.executable,'-m','competition.run_match','--worker',
                 '--submission',str(Path(path).resolve()),'--speed',str(oni_speed),'--output',str(output)]
        if video:command+=['--video',str(Path(video).resolve())]
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        # Allow all 20 full-duration matches even at the permitted 100ms
        # inference budget (3,000 low + 300 high calls per match).
        subprocess.run(command,timeout=7500,check=True,env=env)
        return json.loads(output.read_text())


def tournament(paths):
    reports=[isolated_evaluate(p) for p in paths]
    if len({r['team'] for r in reports})!=len(reports):raise ValueError('Duplicate team names')
    for report in reports:report['rounds']=[report['score_ticks']]
    for speed in (.4,.5):
        groups=defaultdict(list)
        for index,report in enumerate(reports):groups[tuple(report['rounds'])].append(index)
        for indices in groups.values():
            if len(indices)<2:continue
            for index in indices:
                extra=isolated_evaluate(paths[index],speed)
                reports[index].setdefault('tiebreaks',[]).append(extra)
                reports[index]['rounds'].append(extra['score_ticks'])
    reports.sort(key=lambda r:tuple(r['rounds']),reverse=True)
    previous=None;rank=0
    for index,r in enumerate(reports):
        key=tuple(r['rounds'])
        if key!=previous:rank=index+1
        r['rank']=rank;previous=key
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission',type=Path)
    parser.add_argument('--teams-dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path('competition_results.json'))
    parser.add_argument('--video',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--speed',type=float,default=.3,choices=[.3,.4,.5])
    args=parser.parse_args()
    if args.worker:
        if sys.platform=='linux':
            import resource
            resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
        result=evaluate(args.submission,args.speed,args.video)
    elif args.teams_dir:
        paths=sorted(args.teams_dir.glob('*.zip'))
        if not paths:parser.error('No submission ZIP files')
        result=tournament(paths)
    elif args.submission:result=isolated_evaluate(args.submission,args.speed,args.video)
    else:parser.error('--submission or --teams-dir required; legacy model ZIPs are not accepted')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(f'Results: {args.output}',flush=True)


if __name__=='__main__':main()
