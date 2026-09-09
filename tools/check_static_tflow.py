"""Compile every prepared module without importing ML or launching work."""
import ast
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
paths=sorted(set([*root.glob('baselines/tflow*.py'),*root.glob('experiments/tflow/*.py'),
                  root/'experiments/poc_audio/audio_empirical_gain.py',root/'experiments/poc_audio/render_gain_results.py',
                  *root.glob('tests/test_tflow*.py'),*root.glob('tests/test_audio*gain*.py')]))
for path in paths:
    source=path.read_text()
    ast.parse(source,filename=str(path))
    compile(source,str(path),'exec')
manifest=json.loads((root/'configs/tflow/suite_manifest.json').read_text())
configs=list((root/'configs/tflow/prepared').glob('*.json'))
assert len(configs)==len(manifest['benchmarks'])==28
for path in configs:
    cfg=json.loads(path.read_text())
    assert len(cfg['seeds'])==3
    assert cfg['evaluation']['model_evaluations'] in (100,160,512)
    assert cfg['sampler']['t_min']>0
    assert cfg['sampler']['sigma_min']==0.01
result={'status':'passed','compiled_without_execution':len(paths),'configuration_count':len(configs),
        'training_or_benchmark_launched':False,'files':[str(p.relative_to(root)) for p in paths]}
(root/'docs/artifact_audit/static_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
