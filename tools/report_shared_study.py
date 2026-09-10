"""Collect the complete A/B/C/t-Flow study without loading tensors or models.

Partial and failed groups remain visible but never acquire three-seed means.
Historical paper values remain references; no missing outcome is imputed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('A', 'B', 'C', 'tflow')
ANGULAR = ['angular_sw_mean'] + [f'angular_sw_bin{i}' for i in range(4)]
VECTOR = ['radial_w1', 'ks_stat', 'sliced_w1', 'mmd', 'q950_err', 'q990_err', 'q995_err', 'tail_exc_95', 'tail_exc_99'] + ANGULAR
AUDIO = ['digit_acc', 'energy_KS', 'radial_w1', 'cov>q90', 'cov>q95', 'cov>q99', 'cov<q10', 'PIT']
IMAGE = ['fid', 'kid', 'precision', 'recall', 'density', 'coverage', 'radial_w1', 'ks', 'sliced_w1']
RUNTIME = ['sample_time_s', 'total_train_time_s', 'nfe']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flatten_numeric(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten_numeric(child, prefix + '.' + key if prefix else key))
    elif numeric(value):
        result[prefix] = value
    return result


def nonfinite_paths(value, prefix=''):
    if isinstance(value, dict):
        return [p for key, child in value.items() for p in nonfinite_paths(child, prefix + '.' + key if prefix else key)]
    if isinstance(value, list):
        return [p for index, child in enumerate(value) for p in nonfinite_paths(child, f'{prefix}[{index}]')]
    return [prefix] if value is None or (isinstance(value, float) and not math.isfinite(value)) else []


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def summary(values, seeds):
    if len(values) != 3 or len(seeds) != 3 or not all(numeric(x) for x in values):
        raise ValueError('Only three finite final-seed values may be aggregated')
    return {'mean': statistics.mean(values), 'std_population': statistics.pstdev(values),
            'std_sample': statistics.stdev(values), 'n': 3, 'seeds': seeds, 'values': values}


def required_metrics(kind, method):
    return {'vector': VECTOR, 'audio': AUDIO, 'image': IMAGE}[kind] + RUNTIME


def load_cache_manifest(cfg):
    spec = cfg.get('shared_cache', {})
    path, expected = spec.get('manifest_path'), spec.get('manifest_sha256')
    if not path or not expected or not Path(path).is_file():
        return None, 'shared cache manifest not pinned or absent'
    if sha256(path) != expected:
        return None, 'shared cache manifest checksum mismatch'
    value = read_json(path)
    if value.get('status') != 'complete' or value.get('condition_id') != cfg['condition_id']:
        return None, 'shared cache manifest incomplete or wrong condition'
    return value, None


def validate_result(result, cfg, method, seed, manifest):
    issues = []
    if result.get('condition_id') != cfg['condition_id'] or result.get('seed') != seed:
        issues.append('condition/seed identity mismatch')
    if method != 'tflow' and result.get('arm') != method:
        issues.append('arm identity mismatch')
    if method == 'tflow' and (result.get('method') != 'tflow' or result.get('stage') != 'final'):
        issues.append('not a final t-Flow result')
    if result.get('config') != cfg or result.get('config_sha256') != config_hash(cfg):
        issues.append('result/config hash mismatch')
    if result.get('checkpoint', {}).get('step') != cfg['training']['steps']:
        issues.append('prescribed final checkpoint step missing or different')
    if not result.get('implementation_sha256'):
        issues.append('implementation fingerprint missing')
    metrics = result.get('metrics', {})
    for key in required_metrics(cfg['kind'], method):
        if not numeric(metrics.get(key)):
            issues.append('missing/nonfinite required metric: ' + key)
    bad = nonfinite_paths(metrics) + result.get('nonfinite_metric_fields', [])
    if bad:
        issues.append('nonfinite/undefined metrics: ' + ', '.join(sorted(set(bad))))
    sampler = result.get('sampler', {})
    nfe = cfg['evaluation']['model_evaluations']
    if metrics.get('nfe') != nfe or sampler.get('nfe') != nfe:
        issues.append('actual network budget mismatch')
    batches = sampler.get('n_batches')
    if not isinstance(batches, int) or batches < 1 or sampler.get('model_calls_total') != nfe * batches:
        issues.append('actual network call count missing or inconsistent')
    if method != 'tflow':
        if sampler.get('state_renormalization') is not False:
            issues.append('primary A/B/C sampler renormalization policy not verified')
        if not isinstance(result.get('radius_drift'), dict) or nonfinite_paths(result['radius_drift']):
            issues.append('radius drift missing or nonfinite')
        parameters = result.get('parameters', {})
        if not numeric(parameters.get('total_parameters')):
            issues.append('parameter count missing')
    dataset = result.get('dataset', result.get('dataset_manifest'))
    if not isinstance(dataset, dict) or manifest is None:
        issues.append('realized dataset/split compatibility unresolved')
    else:
        for split in ('train', 'val', 'test'):
            expected = manifest['splits'][split]
            if dataset.get('splits', {}).get(split, {}).get('sha256') != expected['values']['sha256_contiguous_bytes']:
                issues.append(split + ' values hash mismatch')
            if dataset.get('split_indices', {}).get(split, {}).get('sha256') != expected['indices']['sha256_contiguous_bytes']:
                issues.append(split + ' indices hash mismatch')
    return issues


def collect_seed(cfg, method, seed, directory, manifest):
    path = directory / 'result.json'
    row = {'seed': seed, 'status': 'missing', 'result_path': str(path), 'metrics': {}, 'issues': []}
    if not path.exists():
        if directory.exists():
            row['status'] = 'in_progress_or_interrupted'
            # A checkpoint alone is not proof that a scheduler job is running.
            row['observed_files'] = sorted(p.name for p in directory.iterdir() if p.is_file())
        return row
    row['result_sha256'] = sha256(path)
    try:
        result = read_json(path)
    except (OSError, ValueError) as error:
        row.update(status='unreadable', issues=[str(error)])
        return row
    row['reported_status'] = result.get('status')
    row['metrics'] = json_safe(result.get('metrics', {}))
    row['hardware'] = result.get('hardware', {})
    row['failure'] = json_safe(result.get('failure'))
    row['nonfinite_metric_fields'] = result.get('nonfinite_metric_fields', [])
    row['implementation_sha256'] = result.get('implementation_sha256')
    row['config_sha256'] = result.get('config_sha256')
    row['checkpoint'] = result.get('checkpoint')
    row['source'] = result.get('source')
    row['dataset_manifest'] = result.get('dataset', result.get('dataset_manifest'))
    row['sampler'] = json_safe(result.get('sampler', {}))
    row['sample_artifact'] = result.get('sample_artifact')
    row['parameters'] = result.get('parameters', {'total_parameters': result.get('metrics', {}).get('n_params')})
    row['radius_drift'] = json_safe(result.get('radius_drift'))
    row['peak_memory'] = result.get('peak_memory', {
        'training': result.get('training_stats', {}).get('peak_memory'),
        'sampling': result.get('sampler', {}).get('peak_memory')})
    row['timing'] = {'kind': 'measured' if result.get('status') == 'complete' else 'recorded_partial_or_failed',
                     'training_scope': result.get('training_stats', {}).get('timing_scope'),
                     'sampling_scope': result.get('sampler', {}).get('timing_scope')}
    row['evaluation_details'] = result.get('evaluation_details', result.get('downstream_evaluation'))
    if result.get('status') != 'complete':
        row['status'] = 'failed' if result.get('status') == 'failed' else 'unrecognized_status'
        return row
    issues = validate_result(result, cfg, method, seed, manifest)
    row.update(status='incompatible' if issues else 'complete', issues=issues)
    audit_path = directory / 'sample_audit.json'
    if row['status'] == 'complete':
        if not audit_path.exists():
            row['status'] = 'awaiting_sample_audit'
        else:
            try:
                audit = read_json(audit_path)
                row['sample_audit'] = audit
                row['sample_audit_sha256'] = sha256(audit_path)
                audit_issues = validate_sample_audit(audit, row, cfg)
                if audit_issues:
                    row['status'] = 'incompatible'
                    row['issues'].extend(audit_issues)
            except (OSError, ValueError) as error:
                row.update(status='incompatible', issues=['Unreadable sample audit: ' + str(error)])
    return row


def validate_sample_audit(audit, row, cfg):
    issues = []
    expected = {'status': 'passed', 'result_sha256': row['result_sha256'],
        'sample_sha256': (row.get('sample_artifact') or {}).get('sha256'),
        'n_samples': cfg['evaluation']['n_samples'], 'dimension': cfg['data']['shape'][1],
        'nfe': cfg['evaluation']['model_evaluations'], 'nonfinite_rows': 0, 'nan_rows': 0, 'inf_rows': 0,
        'configured_sample_seed': cfg['evaluation']['sample_seed']}
    for key, value in expected.items():
        if key not in audit or audit[key] != value:
            issues.append('sample audit mismatch: ' + key)
    if audit.get('errors'):
        issues.append('sample audit recorded errors')
    batches = audit.get('n_batches')
    if not isinstance(batches, int) or batches < 1 or audit.get('model_calls_total') != batches * cfg['evaluation']['model_evaluations']:
        issues.append('sample audit actual network count mismatch')
    classes = None if cfg['kind'] == 'vector' else [cfg['evaluation']['n_samples'] // 10] * 10
    if 'class_counts' not in audit or audit['class_counts'] != classes:
        issues.append('sample audit class-balance mismatch')
    return issues


def aggregate_group(rows, seeds):
    counts = Counter(row['status'] for row in rows)
    group = {'status': 'complete' if counts.get('complete') == 3 else 'incomplete',
             'seed_status_counts': dict(counts), 'seeds': rows, 'aggregate': {},
             'parameter_summary': {}, 'radius_drift_summary': {}, 'memory_summary': {}, 'issues': []}
    if group['status'] != 'complete':
        return group
    for key in ('implementation_sha256', 'dataset_manifest', 'config_sha256', 'source'):
        if any(row.get(key) != rows[0].get(key) for row in rows[1:]):
            group['issues'].append('inconsistent across seeds: ' + key)
    metric_key_sets = [set(flatten_numeric(row['metrics'])) for row in rows]
    if any(keys != metric_key_sets[0] for keys in metric_key_sets[1:]):
        group['issues'].append('metric schema differs across seeds; no fields silently dropped')
    parameter_counts = [row['parameters'].get('total_parameters') for row in rows]
    if len(set(parameter_counts)) != 1:
        group['issues'].append('parameter count differs across seeds')
    if group['issues']:
        group['status'] = 'incompatible'
        return group
    keys = set.intersection(*(set(flatten_numeric(row['metrics'])) for row in rows))
    for key in sorted(keys):
        group['aggregate'][key] = summary([flatten_numeric(row['metrics'])[key] for row in rows], seeds)
    for output, source in [('parameter_summary', 'parameters'), ('radius_drift_summary', 'radius_drift'), ('memory_summary', 'peak_memory')]:
        values = [flatten_numeric(row.get(source)) for row in rows]
        for key in sorted(set.intersection(*(set(v) for v in values))):
            group[output][key] = summary([v[key] for v in values], seeds)
    hardware = [row['hardware'] for row in rows]
    group['hardware_variants'] = list({json.dumps(h, sort_keys=True): h for h in hardware}.values())
    group['timing_comparability_note'] = 'Measured current-run times only. Hardware/software and precision must match for direct runtime claims; historical times are not treated as matched.'
    return group


def paired(groups, left, right, seeds):
    out = {'contrast': left + '-' + right, 'status': 'incomplete', 'metrics': {}, 'per_seed': []}
    if any(groups[arm]['status'] != 'complete' for arm in (left, right)):
        return out
    a, b = groups[left], groups[right]
    for x, y in zip(a['seeds'], b['seeds']):
        if x['seed'] != y['seed'] or x['dataset_manifest'] != y['dataset_manifest'] or x['config_sha256'] != y['config_sha256']:
            out.update(status='incompatible', issue='paired seeds have different data/protocol identity')
            return out
    keys = set(a['aggregate']) & set(b['aggregate'])
    for key in sorted(keys):
        deltas = [flatten_numeric(x['metrics'])[key] - flatten_numeric(y['metrics'])[key] for x, y in zip(a['seeds'], b['seeds'])]
        item = summary(deltas, seeds)
        direction = 'higher' if key == 'digit_acc' else 'lower' if key in VECTOR + ['energy_KS', 'fid', 'kid', 'ks'] else 'not_assigned'
        item['better_direction'] = direction
        item['seeds_favor_left'] = sum(delta > 0 if direction == 'higher' else delta < 0 for delta in deltas) if direction != 'not_assigned' else None
        out['metrics'][key] = item
    for index, seed in enumerate(seeds):
        out['per_seed'].append({'seed': seed, 'deltas': {key: value['values'][index] for key, value in out['metrics'].items()}})
    environment_keys = ('gpu', 'torch', 'python', 'cuda', 'hip')
    matched = []
    for x, y in zip(a['seeds'], b['seeds']):
        hx, hy = x.get('hardware', {}), y.get('hardware', {})
        matched.append(all(key in hx and key in hy and hx[key] == hy[key] for key in environment_keys))
    out['paired_runtime_environment_verified'] = all(matched)
    out['runtime_note'] = 'Current recorded GPU/software fields match for each seed pair' if all(matched) else 'Missing or different hardware/software fields; no matched-runtime claim'
    out['status'] = 'complete'
    return out


def fixed_gain_reference(cfg, cache_manifest, aggregate_path, delivery_path, audio_groups):
    out = {'status': 'unresolved', 'eligible_for_prepared_protocol': False, 'checks': [], 'issues': [],
           'path': str(aggregate_path),
           'interpretation': 'Completed checkpoint-only reference; not original RAFM-Ang arm A. Historical timing is not a matched runtime comparison.',
           'historical_fixed_accuracy': {'mean': 0.810, 'std_population': 0.013, 'source': 'Supplied PDF p26 Table 5'},
           'reproduction_limitation': 'Measured fixed-spherical accuracy is about 0.8067, not an exact reproduction of the reported 0.810 ± 0.013. The original baseline mismatch is preserved.'}
    if cfg is None or cache_manifest is None or not Path(aggregate_path).exists():
        out['issues'].append('Audio prepared config, pinned split manifest, or completed reference absent')
        return out
    try:
        reference = read_json(aggregate_path)
        delivery = read_json(delivery_path)
        pins = {str((ROOT / item['path']).resolve()): item['sha256'] for item in delivery['files']}
        def check(name, actual, expected):
            passed = actual is not None and expected is not None and actual == expected
            recorded_actual = {'json_sha256': config_hash(actual)} if isinstance(actual, dict) else actual
            recorded_expected = {'json_sha256': config_hash(expected)} if isinstance(expected, dict) else expected
            out['checks'].append({'name': name, 'actual': recorded_actual, 'expected': recorded_expected, 'passed': passed})
            if not passed:
                out['issues'].append(name)
        check('archived aggregate checksum', sha256(aggregate_path), pins.get(str(Path(aggregate_path).resolve())))
        for key, spec in [('train_file', cfg['data']['input']), ('test_file', cfg['data']['external_test']), ('classifier', cfg['evaluation']['classifier'])]:
            check(key + ' identity', reference['inputs'][key]['sha256'], spec['sha256'])
        for old, new in [('training_indices_sha256', 'train'), ('internal_validation_indices_sha256', 'val')]:
            check(old, reference['split'][old], cache_manifest['splits'][new]['indices']['sha256_contiguous_bytes'])
        protocol = reference['protocol']
        for name, expected in [('training_seeds', cfg['seeds']), ('n_gen', cfg['evaluation']['n_samples']), ('sample_seed', cfg['evaluation']['sample_seed']), ('model_evaluations', cfg['evaluation']['model_evaluations']), ('class_counts', [200] * 10), ('cfg', 1), ('state_renormalization', False)]:
            check(name, protocol.get(name), expected)
        check('generator training count', reference['split']['training_count'], cfg['data']['split']['n_train'])
        check('internal validation count', reference['split']['internal_validation_count'], cfg['data']['split']['n_val'])
        check('external test count', reference['split']['external_test_count'], cache_manifest['splits']['test']['values']['shape'][0])
        runs = reference.get('runs', [])
        check('three reference seeds', [row.get('seed') for row in runs], cfg['seeds'])
        for run in runs:
            path = Path(aggregate_path).parent / f"seed_{run['seed']}" / 'eval.json'
            check(f"seed {run['seed']} checksum", sha256(path), pins.get(str(path.resolve())))
            check(f"seed {run['seed']} full record", read_json(path), run)
            check(f"seed {run['seed']} invariance", run.get('invariance', {}).get('passed'), True)
            check(f"seed {run['seed']} changed predictions", run.get('invariance', {}).get('prediction_disagreements'), 0)
        out['reference_recorded_status'] = reference.get('status')
        if reference.get('status') not in ('complete', 'baseline_mismatch'):
            out['issues'].append('reference evaluation was not completed')
        out['status'] = 'verified_for_prepared_protocol' if not out['issues'] else 'incompatible'
        out['eligible_for_prepared_protocol'] = not out['issues']
        out['training_seeds'] = reference['protocol']['training_seeds']
        out['measured_full_precision'] = reference['aggregate_full_precision']
        out['measured_archived_rounding'] = reference['aggregate']
        out['hardware'] = reference['hardware']
        out['runtime'] = reference['runtime']
        out['prediction_disagreements'] = sum(row['invariance']['prediction_disagreements'] for row in runs)
        execution = []
        for method, group in (audio_groups or {}).items():
            for row in group['seeds']:
                if row['status'] != 'complete':
                    execution.append({'method': method, 'seed': row['seed'], 'status': 'not_yet_verified'})
                    continue
                sampler = row.get('sample_audit', {})
                required = {'n_samples': 2000, 'configured_sample_seed': 0, 'class_counts': [200] * 10, 'nfe': 160}
                mismatches = [key for key, value in required.items() if sampler.get(key) != value]
                execution.append({'method': method, 'seed': row['seed'], 'status': 'verified' if not mismatches else 'unresolved_or_incompatible', 'unverified_fields': mismatches, 'seed_evidence': sampler.get('seed_evidence', 'Not recorded')})
        out['new_audio_execution_compatibility'] = execution
    except (KeyError, OSError, ValueError) as error:
        out['issues'].append(str(error))
        out['status'] = 'unresolved'
    return out


def auxiliary_status(directory):
    directory = Path(directory)
    if not directory.exists():
        return {'status': 'absent'}
    records = []
    paths = [directory / name for name in ('sanity.json', 'selection.json', 'selected_source.json', 'tuning_history.json', 'failure.json', 'status.json')]
    paths += sorted(directory.glob('candidate_*/failed.json'))
    paths += sorted(directory.glob('candidate_*/validation_step*.json'))
    paths += sorted(directory.glob('*/sanity.json'))
    for path in paths:
        if path.exists():
            try:
                record = read_json(path)
                records.append({'path': str(path), 'sha256': sha256(path), 'record': json_safe(record)})
            except (OSError, ValueError) as error:
                records.append({'path': str(path), 'error': str(error)})
    return {'status': 'records_present' if records else 'directory_present', 'records': records}


def artifact_record(path):
    path = Path(path)
    record = {'path': str(path), 'sha256': sha256(path), 'size_bytes': path.stat().st_size}
    if path.suffix == '.json':
        try:
            record['record'] = json_safe(read_json(path))
        except (OSError, ValueError) as error:
            record['read_error'] = str(error)
    return record


def collect_operations(abc_version, tflow_version):
    """Retain implementation/startup failures separately from final-run outcomes."""
    abc_version, tflow_version = Path(abc_version), Path(tflow_version)
    out = {'implementation_check_failures': [], 'previous_tflow_attempts': [],
           'current_execution_records': {}, 'scheduler_snapshots': [],
           'interpretation': 'Archived implementation/startup failures are not counted as current final scientific seed failures. Scheduler files are dated observations, not live network queries.'}
    for path in sorted((abc_version / 'validation_failures').glob('*/failure.json')):
        entry = artifact_record(path)
        entry['supporting_files'] = [artifact_record(p) for p in sorted(path.parent.iterdir()) if p.is_file() and p != path]
        for name in entry.get('record', {}).get('logs', []):
            log = ROOT / name
            if log.is_file():
                entry['supporting_files'].append(artifact_record(log))
        out['implementation_check_failures'].append(entry)
    out['current_implementation_checks'] = []
    for path in [abc_version / 'unit_checks.xml', abc_version / 'full_backbone_checks.json']:
        if not path.exists():
            continue
        entry = artifact_record(path)
        if path.suffix == '.xml':
            try:
                tree = ET.parse(path).getroot()
                suites = [tree] if tree.tag == 'testsuite' else list(tree.iter('testsuite'))
                entry['counts'] = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
            except (ET.ParseError, ValueError) as error:
                entry['read_error'] = str(error)
        out['current_implementation_checks'].append(entry)
    for previous in sorted(tflow_version.parent.glob('v*')):
        path = previous / 'execution_failure_summary.json'
        if previous == tflow_version or not path.exists():
            continue
        entry = artifact_record(path)
        entry['attempt_root'] = str(previous)
        entry['counts_as_current_final_seed_failure'] = False
        entry['candidate_failure_records'] = [artifact_record(p) for p in sorted((previous / 'tuning').glob('*/candidate_*/failed.json'))]
        entry['execution_records'] = [artifact_record(p) for p in sorted((previous / 'executions').glob('*.json'))]
        out['previous_tflow_attempts'].append(entry)
    for name, version in [('abc', abc_version), ('tflow', tflow_version)]:
        out['current_execution_records'][name] = [artifact_record(p) for p in sorted((version / 'executions').glob('*.json'))]
        for location in (version / 'scheduler_snapshots', version.parent / 'scheduler_snapshots'):
            out['scheduler_snapshots'].extend(artifact_record(p) for p in sorted(location.glob('*.json')))
    out['scheduler_snapshots'] = list({entry['path']: entry for entry in out['scheduler_snapshots']}.values())
    return out


def attach_reference_backend(reference, cfg, audit_path):
    """Keep archival discrepancies separate from measured current-backend reuse."""
    path = Path(audit_path)
    assessment = {'status': 'unresolved', 'verified': False, 'reevaluation_usable': False, 'path': str(path), 'issues': []}
    reference['backend_compatibility'] = assessment
    current = {'status': 'unresolved', 'usable_for_comparison': False, 'audit_path': str(path)}
    reference['current_backend_reference'] = current
    if not path.exists():
        assessment['issues'].append('saved-output evaluation under the new backend has not completed')
        return
    if not isinstance(cfg, dict):
        assessment['issues'].append('prepared audio configuration is absent')
        return
    assessment['sha256'] = sha256(path)
    try:
        audit = read_json(path)
        assessment['record'] = json_safe(audit)
        assessment['audit_status'] = audit.get('status')
        archived = read_json(reference['path'])
        tests = {
            'completed measured audit': audit.get('status') in ('passed', 'compatibility_discrepancy'),
            'verified original data and evaluation protocol': reference.get('eligible_for_prepared_protocol') is True,
            'same shared audio config': audit.get('config_sha256') == config_hash(cfg),
            'full recorded shared audio config': audit.get('config') == cfg,
            'same completed reference': audit.get('aggregate', {}).get('sha256') == sha256(reference['path']),
            'same classifier': audit.get('classifier', {}).get('sha256') == cfg['evaluation']['classifier']['sha256'],
            'same external test': audit.get('external_test', {}).get('sha256') == cfg['data']['external_test']['sha256'],
            'same generator split indices': audit.get('generator_split_indices_match') is True,
            'zero old-versus-new changed predictions': audit.get('totals', {}).get('old_vs_new_prediction_disagreements') == 0,
            'zero postrescale changed predictions': audit.get('totals', {}).get('postrescale_prediction_disagreements') == 0,
            'all original seeds': [row.get('seed') for row in audit.get('runs', [])] == cfg['seeds'],
            'saved outputs only': audit.get('generation_performed') is False and audit.get('training_updates') == 0,
        }
        sources = ('tools/check_audio_reference_backend.py', 'experiments/rafm_inputs/run.py', 'experiments/tflow/run.py',
                   'rafm/utils/seeds.py', 'experiments/poc_audio/audio_classifier.py', 'experiments/poc_audio/audio_empirical_gain.py')
        for name in sources:
            tests['same source ' + name] = audit.get('source_sha256', {}).get(name) == sha256(ROOT / name)
        verified = audit.get('verification', [])
        tests['all artifact verifications passed'] = bool(verified) and all(item.get('status') == 'passed' for item in verified)
        verified_pins = {(item.get('path'), item.get('sha256')) for item in verified if item.get('status') == 'passed'}
        expected_inputs = [cfg['data']['input'], cfg['data']['external_test'], cfg['data']['split']['file'], cfg['evaluation']['classifier']]
        for spec in expected_inputs:
            tests['verified input ' + spec['path']] = (spec['path'], spec['sha256']) in verified_pins
        archived_runs = {row['seed']: row for row in archived['runs']}
        primary = ('digit_acc', 'energy_KS', 'cov>q90', 'cov>q95', 'cov>q99', 'cov<q10')
        energy_names = {'energy_KS': 'ks', 'cov>q90': 'cov_gt_q90', 'cov>q95': 'cov_gt_q95',
                        'cov>q99': 'cov_gt_q99', 'cov<q10': 'cov_lt_q10', 'PIT': 'pit_mean', 'radial_w1': 'radial_w1'}
        differences = []
        for row in audit.get('runs', []):
            prefix = f"seed {row.get('seed')} "
            tests[prefix + 'sample count'] = row.get('n') == cfg['evaluation']['n_samples']
            tests[prefix + 'gain invariance'] = row.get('invariance', {}).get('passed') is True and row.get('invariance', {}).get('prediction_disagreements') == 0
            original = archived_runs[row['seed']]
            sample = original['samples']
            tests[prefix + 'same saved samples'] = row.get('input_samples') == sample and (sample['path'], sample['sha256']) in verified_pins
            for version in ('baseline', 'posthoc'):
                comparison = row.get(version, {}).get('comparison', {})
                tests[prefix + version + ' predictions'] = comparison.get('prediction_disagreements_vs_cached') == 0
                measured = row.get(version, {}).get('metrics', {})
                old = comparison.get('old_metrics', {})
                deltas = comparison.get('metric_deltas', {})
                unrounded = original[version]['unrounded']
                expected_old = {'digit_acc': unrounded['digit_acc'], **{key: unrounded['energy'][value] for key, value in energy_names.items()}}
                tests[prefix + version + ' archived metrics identity'] = old == expected_old
                tests[prefix + version + ' all current metrics finite'] = all(numeric(measured.get(key)) for key in AUDIO)
                tests[prefix + version + ' unchanged accuracy KS and coverage'] = all(measured.get(key) == old.get(key) and deltas.get(key) == 0 for key in primary)
                tests[prefix + version + ' measured deltas consistent'] = all(numeric(deltas.get(key)) and deltas[key] == measured[key] - old[key] for key in AUDIO if key in measured and key in old) and set(AUDIO).issubset(deltas)
                differences.append({'seed': row['seed'], 'version': version, 'metric_deltas': deltas,
                                    'all_unrounded_energy_deltas': comparison.get('all_unrounded_energy_deltas'),
                                    'maximum_logit_absolute_difference': comparison.get('maximum_logit_absolute_difference'),
                                    'energy_bin_accuracies_exactly_equal': comparison.get('energy_bin_accuracies_exactly_equal'),
                                    'energy_ks_tail_metrics_exactly_equal': comparison.get('energy_ks_tail_metrics_exactly_equal')})
        for version in ('baseline', 'posthoc'):
            for key in AUDIO:
                values = [row.get(version, {}).get('metrics', {}).get(key) for row in audit.get('runs', [])]
                stats = audit.get('aggregate_metrics', {}).get(version, {}).get(key, {})
                tests[version + ' measured aggregate ' + key] = len(values) == 3 and all(numeric(value) for value in values) and stats == {
                    'mean': statistics.mean(values), 'std_population': statistics.pstdev(values), 'values': values, 'n': 3}
        assessment['checks'] = tests
        assessment['issues'] = [name for name, passed in tests.items() if not passed]
        assessment['reevaluation_usable'] = not assessment['issues']
        assessment['verified'] = not assessment['issues'] and audit.get('status') == 'passed' and all(
            row[version]['comparison'].get('energy_ks_tail_metrics_exactly_equal') is True
            for row in audit['runs'] for version in ('baseline', 'posthoc'))
        assessment['status'] = 'measured_compatible' if assessment['verified'] else 'compatibility_discrepancy_or_missing_evidence'
        if assessment['reevaluation_usable']:
            current.update(status='measured_under_current_backend_with_recorded_differences' if audit['status'] != 'passed' else 'measured_under_current_backend',
                usable_for_comparison=True, audit_sha256=assessment['sha256'], aggregate_metrics=audit['aggregate_metrics'],
                runs=audit['runs'], differences=differences, backend_flags=audit.get('backend_flags'), hardware=audit.get('hardware'),
                interpretation='These are newly measured metrics for the same saved Y/X under the current study backend. Archived metrics and raw audit status remain unchanged. Digit predictions, accuracy, energy KS and coverage rates are identical; PIT, radial W1, logits and near-fixed-radius energy-bin assignments can differ. Only these current-backend values enter new comparisons; no historical training-time comparison.')
    except (OSError, ValueError, KeyError, TypeError) as error:
        assessment['issues'].append(str(error))


def audio_comparison_rows(report):
    """Reference rows remain separate from the expected A/B/C/t-Flow seed count."""
    metrics = ('digit_acc', 'energy_KS', 'cov>q95', 'cov>q99')
    audio = next((item for item in report['conditions'] if item['condition_id'] == 'audiomnist_stft'), None)
    rows = []
    if audio:
        for method, group in audio['methods'].items():
            rows.append({'method': method, 'status': group['status'], 'role': 'new matched study arm',
                         'metrics': {key: group['aggregate'][key] for key in metrics if key in group['aggregate']}})
    ref = report['fixed_spherical_gain_reference']
    current = ref.get('current_backend_reference', {})
    if ref.get('eligible_for_prepared_protocol') and current.get('usable_for_comparison') is True:
        rows.append({'method': 'fixed_spherical_empirical_gain_reference', 'status': current['status'],
                     'role': 'checkpoint reference re-evaluated under current backend, not new A or historical paper value',
                     'audit_sha256': current['audit_sha256'],
                     'metrics': {key: dict(current['aggregate_metrics']['posthoc'][key], seeds=ref['training_seeds']) for key in metrics}})
    return rows


def collect(config_root, abc_root, tflow_root, gain_path, gain_delivery):
    config_root, abc_root, tflow_root = map(Path, (config_root, abc_root, tflow_root))
    plan = read_json(config_root / 'plan.json')
    conditions = sorted(set(plan['materializable_conditions']) | set(plan['blocked_conditions']))
    if len(conditions) != plan['complete_scope_conditions']:
        raise ValueError('Plan completeness does not match listed conditions')
    report = {'schema_version': 1, 'study_id': plan['study_id'], 'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'status': 'incomplete', 'expected_final_runs': len(conditions) * 12,
              'aggregation_policy': 'Only all three prescribed final seeds, complete finite metrics and matching config/data. Population SD is primary; sample SD is separately labelled. No survivor averages.',
              'plan': {'path': str(config_root / 'plan.json'), 'sha256': sha256(config_root / 'plan.json')},
              'conditions': [], 'input_files': []}
    cache_by_condition = {}
    cfg_by_condition = {}
    all_rows = []
    for condition in conditions:
        prepared = config_root / 'prepared' / (condition + '.json')
        cfg_path = prepared if prepared.exists() else config_root / 'drafts' / (condition + '.json')
        cfg = read_json(cfg_path)
        cfg_by_condition[condition] = cfg
        cache, cache_issue = load_cache_manifest(cfg)
        cache_by_condition[condition] = cache
        item = {'condition_id': condition, 'kind': cfg['kind'], 'seeds': cfg['seeds'],
                'comparison_label': cfg['provenance']['comparison_label'], 'config_path': str(cfg_path),
                'config_sha256': config_hash(cfg), 'blocking_issues': plan['blocked_conditions'].get(condition, []),
                'cache_status': 'verified_manifest' if cache else 'unresolved', 'cache_issue': cache_issue,
                'training': cfg['training'], 'evaluation': cfg['evaluation'], 'methods': {}}
        report['input_files'].append({'path': str(cfg_path), 'sha256': sha256(cfg_path)})
        for method in METHODS:
            base = tflow_root / condition if method == 'tflow' else abc_root / condition / method
            rows = [collect_seed(cfg, method, seed, base / f'seed_{seed}', cache) for seed in cfg['seeds']]
            if item['blocking_issues']:
                for row in rows:
                    if row['status'] == 'missing':
                        row['status'] = 'blocked'
                    elif row['status'] == 'complete':
                        row['status'] = 'incompatible'
                        row['issues'].append('Condition still has unresolved protocol blockers')
            item['methods'][method] = aggregate_group(rows, cfg['seeds'])
            all_rows.extend(rows)
        item['paired_contrasts'] = {left + '-' + right: paired(item['methods'], left, right, cfg['seeds']) for left, right in [('B', 'A'), ('B', 'C')]}
        b, c = item['methods']['B'], item['methods']['C']
        item['B_C_parameter_equality'] = 'pending'
        if b['status'] == c['status'] == 'complete':
            item['B_C_parameter_equality'] = 'passed' if b['parameter_summary'].get('total_parameters') == c['parameter_summary'].get('total_parameters') else 'failed'
            if item['B_C_parameter_equality'] == 'failed':
                item['paired_contrasts']['B-C'] = {'status': 'incompatible', 'issue': 'B/C parameter counts differ', 'metrics': {}}
        item['sanity'] = {'abc': auxiliary_status(abc_root.parent / 'sanity' / condition), 'tflow': auxiliary_status(tflow_root.parent / 'sanity' / condition)}
        item['tflow_tuning'] = auxiliary_status(tflow_root.parent / 'tuning' / condition)
        report['conditions'].append(item)
    counts = Counter(row['status'] for row in all_rows)
    report['seed_status_counts'] = dict(counts)
    report['complete_method_groups'] = sum(group['status'] == 'complete' for item in report['conditions'] for group in item['methods'].values())
    report['complete_conditions'] = [item['condition_id'] for item in report['conditions'] if all(g['status'] == 'complete' for g in item['methods'].values()) and item['B_C_parameter_equality'] == 'passed']
    if counts.get('complete') == report['expected_final_runs'] and len(report['complete_conditions']) == len(conditions):
        report['status'] = 'complete'
    audio = next((item for item in report['conditions'] if item['condition_id'] == 'audiomnist_stft'), None)
    report['operations'] = collect_operations(abc_root.parent, tflow_root.parent)
    report['fixed_spherical_gain_reference'] = fixed_gain_reference(cfg_by_condition.get('audiomnist_stft'), cache_by_condition.get('audiomnist_stft'), gain_path, gain_delivery, audio['methods'] if audio else None)
    attach_reference_backend(report['fixed_spherical_gain_reference'], cfg_by_condition.get('audiomnist_stft'), abc_root.parent / 'audio_reference_backend_check.json')
    report['audio_comparison_rows'] = audio_comparison_rows(report)
    return report


def fmt(group, metric):
    entry = group.get('aggregate', {}).get(metric)
    return '—' if entry is None else f"{entry['mean']:.6g} ± {entry['std_population']:.3g}"


def write_outputs(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'report.json', report)
    lines = ['# RAFM-Ang input parameterization and t-Flow study', '',
             f"Status: **{report['status']}**. {report['seed_status_counts'].get('complete', 0)}/{report['expected_final_runs']} final seeds verified; {report['complete_method_groups']}/112 three-seed method groups complete.", '',
             'Three-seed means use population standard deviation. Missing, failed and incompatible groups have no mean. Training directories do not establish that a scheduler job is currently running.', '',
             '| Condition | Comparison | A | B | C | t-Flow | Blockers |', '|---|---|---|---|---|---|---|']
    for item in report['conditions']:
        cells = [item['condition_id'], item['comparison_label']] + [f"{g['seed_status_counts'].get('complete', 0)}/3 {g['status']}" for g in item['methods'].values()] + [', '.join(item['blocking_issues']) or '—']
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Failures and compatibility issues', '']
    failed = 0
    for item in report['conditions']:
        for method, group in item['methods'].items():
            for row in group['seeds']:
                if row['status'] in ('failed', 'incompatible', 'unreadable', 'unrecognized_status'):
                    failed += 1
                    reason = '; '.join(row.get('issues', [])) or json.dumps(row.get('failure') or row.get('nonfinite_metric_fields'))
                    lines.append(f"- {item['condition_id']} / {method} / seed {row['seed']}: {row['status']}; {reason}; {row['result_path']}")
    if not failed:
        lines.append('No final-run failure records are present in this snapshot. This does not imply missing runs passed.')
    operations = report.get('operations', {})
    lines += ['', '## Preserved implementation and startup failures', '']
    for entry in operations.get('implementation_check_failures', []):
        data = entry.get('record', {})
        lines.append(f"- Implementation-check job {data.get('job_id')}: {data.get('passed')} passed / {data.get('failed')} failed; {data.get('error')}. Resolution: {data.get('resolution')}. Benchmark results affected: {data.get('training_or_benchmark_results_affected')}.")
    for entry in operations.get('previous_tflow_attempts', []):
        data = entry.get('record', {})
        lines.append(f"- Earlier t-Flow attempt {entry['attempt_root']}: {data.get('status')}; {len(data.get('failures', []))} startup candidate failures, {data.get('optimizer_updates_observed')} observed optimizer updates, {len(data.get('checkpoint_files', []))} checkpoints. These cancelled attempts are separate from current scientific seed outcomes. Resolution: {data.get('resolution')}.")
    if not operations.get('implementation_check_failures') and not operations.get('previous_tflow_attempts'):
        lines.append('No separate implementation/startup failure archives found.')
    lines += ['', 'Current check results and dated worker/scheduler snapshots are preserved in report.json; this collector makes no live scheduler queries.']
    lines += ['', '## Measured comparison', '', '| Condition | Method | Primary metric | Mean ± population SD | Angular SW | Train seconds | Sample seconds |', '|---|---|---|---|---|---|---|']
    for item in report['conditions']:
        primary = 'digit_acc' if item['kind'] == 'audio' else 'fid' if item['kind'] == 'image' else 'sliced_w1'
        for method, group in item['methods'].items():
            lines.append('| ' + ' | '.join([item['condition_id'], method, primary, fmt(group, primary), fmt(group, 'angular_sw_mean'), fmt(group, 'total_train_time_s'), fmt(group, 'sample_time_s')]) + ' |')
    ref = report['fixed_spherical_gain_reference']
    lines += ['', '## Fixed-spherical + gain audio reference', '', 'Compatibility: **' + ref['status'] + '**. ' + ref['reproduction_limitation']]
    if ref.get('eligible_for_prepared_protocol'):
        acc = ref['measured_full_precision']['digit_acc']
        lines += ['', f"Measured reference accuracy {acc['mean']:.7f} ± {acc['std']:.7f} (population SD), energy KS {ref['measured_full_precision']['energy_KS']['mean']:.7f}; {ref['prediction_disagreements']} changed digit predictions across 6,000 paired outputs. New-run execution compatibility remains separately listed in report.json."]
    backend = ref.get('backend_compatibility', {})
    current = ref.get('current_backend_reference', {})
    lines += ['', 'Raw new-evaluator audit status: **' + str(backend.get('audit_status', 'unresolved')) + '**. Current measured reference: **' + current.get('status', 'unresolved') + '**.']
    if current.get('usable_for_comparison'):
        lines += ['', current['interpretation'], '', 'Measured current-minus-archived differences (each seed/version):']
        for item in current['differences']:
            deltas = {key: value for key, value in item['metric_deltas'].items() if value != 0}
            lines.append('- Seed ' + str(item['seed']) + ' / ' + item['version'] + ': ' + json.dumps(deltas, sort_keys=True) + '; energy-bin accuracies identical: ' + str(item['energy_bin_accuracies_exactly_equal']) + '.')
    lines += ['', '| Audio method | Role / status | Digit accuracy | Energy KS | Coverage > q95 | Coverage > q99 |', '|---|---|---|---|---|---|']
    for row in report.get('audio_comparison_rows', []):
        cells = [row['method'], row['role'] + ' / ' + row['status']]
        for key in ('digit_acc', 'energy_KS', 'cov>q95', 'cov>q99'):
            value = row['metrics'].get(key)
            cells.append('—' if value is None else f"{value['mean']:.7f} ± {value['std_population']:.7f}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    if ref['issues']:
        lines += ['', 'Unresolved checks: ' + '; '.join(ref['issues'])]
    lines += ['', '## Findings status', '',
              '1. Whether normalized input improves original RAFM-Ang: use the paired B−A deltas for complete conditions; no full-suite conclusion before completion.',
              '2. Whether radius conditioning helps: use paired B−C deltas, retaining identical parameter counts. Audio gains were constructed independently of content; a benefit is not assumed.',
              '3. Consistency: each paired metric records all three deltas and the number favoring B. No significance claim follows from three seeds alone.',
              '4. Cost: parameter, conditioning overhead, measured training/sampling time and recorded memory are reported per seed. Missing memory fields remain absent; historical timing is not treated as matched.', '',
              'See report.json for all errors, source/config fingerprints, dataset identities, hardware, tuning/sanity records, radius drift, and paired deltas.']
    (directory / 'README.md').write_text('\n'.join(lines) + '\n')
    with (directory / 'per_seed_metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['condition', 'method', 'seed', 'status', 'metric', 'value', 'result_path'])
        for item in report['conditions']:
            for method, group in item['methods'].items():
                for row in group['seeds']:
                    metrics = flatten_numeric(row['metrics'])
                    for key, value in sorted(metrics.items()) if metrics else [('', '')]:
                        writer.writerow([item['condition_id'], method, row['seed'], row['status'], key, value, row['result_path']])
    with (directory / 'aggregate_metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['condition', 'method', 'status', 'metric', 'mean', 'std_population', 'std_sample', 'n'])
        for item in report['conditions']:
            for method, group in item['methods'].items():
                for key, value in sorted(group['aggregate'].items()) if group['aggregate'] else [('', {})]:
                    writer.writerow([item['condition_id'], method, group['status'], key] + [value.get(k, '') for k in ('mean', 'std_population', 'std_sample', 'n')])
    with (directory / 'paired_deltas.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['condition', 'contrast', 'status', 'metric', 'seed', 'delta', 'mean_delta', 'std_population', 'seeds_favor_B'])
        for item in report['conditions']:
            for name, contrast in item['paired_contrasts'].items():
                for key, value in sorted(contrast['metrics'].items()) if contrast['metrics'] else [('', {})]:
                    for seed, delta in zip(value.get('seeds', ['']), value.get('values', [''])):
                        writer.writerow([item['condition_id'], name, contrast['status'], key, seed, delta, value.get('mean', ''), value.get('std_population', ''), value.get('seeds_favor_left', '')])
    latex = ['% Standalone measured-results addition; published tables unchanged.', r'\begin{tabular}{lll}', r'\toprule', r'Condition / method & Primary metric & Mean $\pm$ population SD \\', r'\midrule']
    for item in report['conditions']:
        metric = 'digit_acc' if item['kind'] == 'audio' else 'fid' if item['kind'] == 'image' else 'sliced_w1'
        for method, group in item['methods'].items():
            value = fmt(group, metric).replace('—', '--').replace(' ± ', r' $\pm$ ')
            latex.append(item['condition_id'].replace('_', r'\_') + ' / ' + method + ' & ' + metric.replace('_', r'\_') + ' & ' + value + r' \\')
    latex += [r'\bottomrule', r'\end{tabular}', '% -- denotes incomplete/failed/blocked, never a zero measurement.']
    (directory / 'comparison_table.tex').write_text('\n'.join(latex) + '\n')
    text = (f"The input-parameterization study currently has {len(report['complete_conditions'])} fully measured conditions out of 28. "
            "Its A/B/C comparisons preserve the matched-radius spherical objective and ambient sampler; B and C differ only in whether the shared scalar conditioning pathway receives standardized training-fit log-radius or zero. "
            "The added t-Flow comparison uses the published noise-prediction formulation at matched per-method final training budgets, with its separately reported validation-only tuning compute. "
            "Twenty synthetic conditions use new explicitly seeded caches shared across the compared methods; their results must not replace historical realizations. "
            "PIV d32 and image split/reference provenance remain listed as blockers until resolved. "
            "No improvement, theoretical guarantee, or full-suite ranking is asserted before observing the complete results. "
            "The completed audio fixed-spherical+empirical-gain control is a separate reference exploiting the constructed independence of gain and content. Its measured accuracy is approximately 0.8067 rather than the historical 0.810, a preserved reproduction discrepancy. "
            "This control does not replace the separate RAFM-Ang versus RAFM-Vel interpretation.\n")
    (directory / 'manuscript_addition.md').write_text(text)
    audio_rows = report.get('audio_comparison_rows', [])
    with (directory / 'audio_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['method', 'role', 'status', 'metric', 'mean', 'std_population', 'n'])
        for row in audio_rows:
            for key in ('digit_acc', 'energy_KS', 'cov>q95', 'cov>q99'):
                value = row['metrics'].get(key, {})
                writer.writerow([row['method'], row['role'], row['status'], key, value.get('mean', ''), value.get('std_population', ''), value.get('n', '')])
    audio_tex = [r'\begin{tabular}{lrrrr}', r'\toprule', r'Method & Digit accuracy & Energy KS & $>q_{95}$ & $>q_{99}$ \\', r'\midrule']
    for row in audio_rows:
        title = 'Fixed-spherical + gain (checkpoint reference)' if row['method'] == 'fixed_spherical_empirical_gain_reference' else row['method']
        cells = [title]
        for key in ('digit_acc', 'energy_KS', 'cov>q95', 'cov>q99'):
            value = row['metrics'].get(key)
            cells.append('--' if value is None else f"{value['mean']:.7f} $\\pm$ {value['std_population']:.7f}")
        audio_tex.append(' & '.join(cells) + r' \\')
    audio_tex += [r'\bottomrule', r'\end{tabular}', '% Measured checkpoint reference ~0.8067 accuracy is not historical paper 0.810 +/- 0.013.', '% All errors are population SD; -- means incomplete, never zero. No historical training-time comparison.']
    (directory / 'audio_comparison.tex').write_text('\n'.join(audio_tex) + '\n')
    assets = [p for p in directory.iterdir() if p.is_file() and p.name != 'report_files.json']
    write_json(directory / 'report_files.json', {'files': [{'path': str(p), 'sha256': sha256(p)} for p in sorted(assets)]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-root', type=Path, default=ROOT / 'configs/rafm_input_study')
    parser.add_argument('--abc-root', type=Path, default=ROOT / 'outputs_rafm_input_study/v1/final')
    parser.add_argument('--tflow-root', type=Path, default=ROOT / 'outputs_tflow_full/v2/final')
    parser.add_argument('--gain-result', type=Path, default=ROOT / 'outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/aggregate.json')
    parser.add_argument('--gain-delivery', type=Path, default=ROOT / 'outputs_audio_gain/delivery_manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs_rafm_input_study/v1/report')
    parser.add_argument('--plots', action='store_true', help='Render measured figures on a compute node; skips cleanly before any triplet completes')
    args = parser.parse_args()
    report = collect(args.config_root, args.abc_root, args.tflow_root, args.gain_result, args.gain_delivery)
    write_outputs(report, args.output)
    if args.plots:
        from plot_shared_study import plot_report
        plot_report(report, args.output / 'figures')
    print(json.dumps({'status': report['status'], 'seed_status_counts': report['seed_status_counts'],
                      'complete_method_groups': report['complete_method_groups'], 'reference': report['fixed_spherical_gain_reference']['status'],
                      'report': str(args.output / 'report.json')}, indent=2))


if __name__ == '__main__':
    main()
