"""Run the original tiny t-Flow sanity on one verified recovered condition."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True, choices=('imagenette_dcae', 'piv_d32'))
    args = parser.parse_args()
    path = ROOT / 'configs/rafm_input_study_recovered/prepared' / (args.condition + '.json')
    preparation = json.loads((ROOT / 'configs/rafm_input_study_recovered/materialization.json').read_text())
    row = next(row for row in preparation['rows'] if row['condition_id'] == args.condition)
    if row['status'] != 'verified' or hashlib.sha256(path.read_bytes()).hexdigest() != row['config']['sha256']:
        raise ValueError('Recovered configuration does not match the successful artifact audit')
    from tools.experiment_entrypoint import initialize_accelerator
    initialize_accelerator()
    from experiments.tflow.sanity import check
    print(json.dumps(check(json.loads(path.read_text()), ROOT / 'outputs_tflow_full/v1/sanity'), indent=2), flush=True)


if __name__ == '__main__':
    main()
