"""Read-only hashes of existing decoder/reference resources; no ML imports."""
import hashlib
import json
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


root=Path(__file__).resolve().parents[1]
core=root.parent/'msgm-sparse-control'
decoder=core/'.cache/huggingface/hub/models--mit-han-lab--dc-ae-f32c32-sana-1.0-diffusers/snapshots/ca69e17e97609e64ce055115a6515215109b1f50'
weights=core/'.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'
manifest=core/'real_ref_manifest.sha256'
records=[line.split('  ',1) for line in manifest.read_text().splitlines() if line and not line.startswith('#')]
assert len(records)==3925
directory_digest=hashlib.sha256()
for checksum,name in sorted(records,key=lambda row:row[1]):
    directory_digest.update((name+'\0'+checksum+'\n').encode())
resources={'decoder_path':str(decoder),'decoder_files_sha256':{name:digest(decoder/name) for name in ('config.json','diffusion_pytorch_model.safetensors')},
           'inception_weights_path':str(weights),'inception_weights_sha256':digest(weights),
           'expected_reference_sha256':directory_digest.hexdigest(),'real_n':3925,
           'resource_status':'existing_local_files_hashed_without_model_load',
           'split_status':'still_blocked_requires_original_indices_and_reference_disjointness'}
(root/'configs/tflow/image_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
print(json.dumps(resources,indent=2))
