"""Report the 252 prescribed RAFM A/B/C runs, independently of t-Flow completion.

Standard-library only: reads configs, result JSONs and validation receipts; never
loads models/tensors, samples, trains, or queries/submits scheduler jobs. Existing
collectors and their output files are left untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import report_shared_study as shared
import summarize_shared_study as findings

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'configs/rafm_input_study_complete'
OUTPUT = ROOT / 'outputs_rafm_input_study/complete_report'
ARMS = ('A', 'B', 'C')
TERMINAL = {'complete', 'failed'}
FILES = ('abc_completion.json', 'abc_completion.md', 'abc_seed_status.csv',
         'abc_per_seed_metrics.csv', 'abc_aggregate_metrics.csv',
         'abc_tflow_reference_metrics.csv')


def check_scope(report, config_root):
    """Do not let a 26-condition snapshot masquerade as the recovered suite."""
    plan_path = config_root / 'plan.json'
    plan = shared.read_json(plan_path)
    names = sorted(set(plan['materializable_conditions']) | set(plan['blocked_conditions']))
    items = report.get('conditions', [])
    if len(names) != 28 or plan.get('complete_scope_conditions') != 28:
        raise ValueError('Completion report requires the complete 28-condition plan')
    if len(items) != 28 or sorted(item['condition_id'] for item in items) != names:
        raise ValueError('Source report does not contain exactly all 28 conditions')
    if report.get('plan', {}).get('sha256') != shared.sha256(plan_path):
        raise ValueError('Source report plan hash differs from the current complete plan')
    configs = {}
    for item in items:
        name = item['condition_id']
        path = config_root / 'prepared' / (name + '.json')
        cfg = shared.read_json(path)
        if cfg['condition_id'] != name or item.get('config_sha256') != shared.config_hash(cfg):
            raise ValueError('Source report/config identity differs: ' + name)
        expected = [8925, 1234, 7] if cfg['kind'] in ('audio', 'image') else [8925, 77395, 65457]
        if cfg['seeds'] != expected or item['seeds'] != expected:
            raise ValueError('Prescribed three-seed identity differs: ' + name)
        for method in (*ARMS, 'tflow'):
            rows = item['methods'][method]['seeds']
            if len(rows) != 3 or [row['seed'] for row in rows] != expected:
                raise ValueError('Duplicate, missing or reordered seed rows: ' + name + '/' + method)
            group = item['methods'][method]
            if group.get('status') == 'complete':
                if any(row.get('status') != 'complete' for row in rows):
                    raise ValueError('Complete group contains an unverified seed: ' + name + '/' + method)
            elif group.get('aggregate'):
                raise ValueError('Incomplete group has a prohibited survivor aggregate: ' + name + '/' + method)
        configs[name] = cfg
    return plan, configs


def status_summary(groups, expected):
    groups = list(groups)
    rows = [row for group in groups for row in group['seeds']]
    if len(rows) != expected:
        raise ValueError('Expected-run count differs from seed inventory')
    counts = Counter(row['status'] for row in rows)
    incompatible_groups = sum(group['status'] == 'incompatible' for group in groups)
    if counts['complete'] == expected and not incompatible_groups:
        status = 'complete'
    elif all(row['status'] in TERMINAL for row in rows) and not incompatible_groups:
        status = 'finished_with_failures'
    else:
        status = 'incomplete'
    return {'status': status, 'expected_runs': expected, 'seed_status_counts': dict(counts),
            'complete_runs': counts['complete'], 'failed_runs': counts['failed'],
            'missing_runs': counts['missing'],
            'in_progress_or_interrupted_runs': counts['in_progress_or_interrupted'],
            'awaiting_sample_audit_runs': counts['awaiting_sample_audit'],
            'other_unresolved_runs': sum(n for key, n in counts.items()
                if key not in TERMINAL | {'missing', 'in_progress_or_interrupted', 'awaiting_sample_audit'}),
            'complete_three_seed_groups': sum(group['status'] == 'complete' for group in groups),
            'incompatible_three_seed_groups': incompatible_groups,
            'all_prescribed_outcomes_recorded': all(row['status'] in TERMINAL for row in rows),
            'all_metrics_successful': counts['complete'] == expected and not incompatible_groups}


def build_report(source, config_root, source_record):
    plan, configs = check_scope(source, config_root)
    now = datetime.now(timezone.utc)
    result = {'schema_version': 1, 'study_id': source['study_id'],
        'generated_at_utc': now.isoformat(), 'generated_at_paris': now.astimezone(ZoneInfo('Europe/Paris')).isoformat(),
        'source_report': source_record, 'source_results_at_utc': source['generated_at_utc'],
        'scope': '28 conditions × three RAFM input arms × three training seeds = 252 prescribed final runs',
        'methods': {'A': 'Original RAFM-Ang', 'B': 'Unit-direction input plus standardized log-radius',
                    'C': 'Unit-direction input with constant-zero radius condition'},
        'aggregation_policy': 'Mean and population SD only for all three compatible audited final seeds. '
            'Partial and failed groups retain per-seed values but have no aggregate.',
        'scheduler_note': 'No live scheduler query. A training directory without final result is labelled '
            'in_progress_or_interrupted; it does not establish that a job is running.',
        'implementation': {name: shared.sha256(ROOT / name) for name in
            ('tools/report_rafm_completion.py', 'tools/report_shared_study.py', 'tools/summarize_shared_study.py')},
        'plan': source['plan'], 'conditions': [], 'contrasts': {'B-A': [], 'B-C': []},
        'new_shared_synthetic_conditions': plan.get('new_synthetic_realizations', []),
        'limitations': [
            'New explicitly seeded synthetic caches are shared by the new methods; they are not recovered historical realizations.',
            'Three seeds and differences in observed means alone do not establish statistical significance.',
            'Measured runtime comparisons require the same recorded hardware/software/precision and retain timing-scope limitations.',
            'The existing t-Flow adaptation has documented failures and sampler/backbone limitations; it is a qualified reference, not an ABC completion requirement.',
            'The inherited RAFM near-antipodal path routine can produce excessively large finite angular targets. '
            'Toy sampling failures remain scientific failures; the audit identifies a numerical defect but does not '
            'attribute each failed run to a specific unsaved training example or integration step.',
        ],
        'failure_analysis_documents': [{'path': str(ROOT / name), 'sha256': shared.sha256(ROOT / name)}
            for name in ('docs/rafm_toy_numerical_failure_analysis.md',
                         'docs/tflow_matched_backbone_failure_analysis.md')],
        'fixed_spherical_gain_reference': source.get('fixed_spherical_gain_reference', {}),
        'audio_comparison_rows': source.get('audio_comparison_rows', []),
    }
    for item in source['conditions']:
        cfg = configs[item['condition_id']]
        recovered = cfg['provenance'].get('recovered_artifacts', {})
        condition = {key: item[key] for key in ('condition_id', 'kind', 'seeds', 'comparison_label',
            'config_path', 'config_sha256', 'blocking_issues', 'cache_status', 'cache_issue',
            'training', 'evaluation', 'B_C_parameter_equality')}
        condition.update(methods={arm: item['methods'][arm] for arm in ARMS},
            tflow_reference=item['methods']['tflow'],
            provenance_notes={'historical_result_reuse': cfg['provenance'].get('historical_result_reuse'),
                'recovered_commit_caveat': recovered.get('commit_caveat'),
                'image_reference_overlap': recovered.get('numerical_verification', {}).get('fid_reference_overlap')},
            parameter_cost=findings.parameter_cost(item),
            paired_contrasts={name: findings.comparison(item, name[0], name[2]) for name in ('B-A', 'B-C')})
        result['conditions'].append(condition)
        for name in result['contrasts']:
            result['contrasts'][name].append(condition['paired_contrasts'][name])
    result['abc'] = status_summary((g for c in result['conditions'] for g in c['methods'].values()), 252)
    result['status'] = result['abc']['status']
    result['tflow_reference'] = status_summary((c['tflow_reference'] for c in result['conditions']), 84)
    result['tflow_reference'].update(required_for_abc_completion=False,
        continuation_scope='Existing results only; no new recovered-condition t-Flow tuning or training launched by this reporter',
        missing_is_not_failure=True)
    result['findings'] = {name: findings.outcome_tally(rows) for name, rows in result['contrasts'].items()}
    result['complete_abc_conditions'] = [c['condition_id'] for c in result['conditions']
        if all(g['status'] == 'complete' for g in c['methods'].values()) and c['B_C_parameter_equality'] == 'passed']
    return result


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def csv_text(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def cell(group, metric):
    value = group.get('aggregate', {}).get(metric)
    if group['status'] != 'complete' or value is None:
        counts = ', '.join(f'{n} {status}' for status, n in sorted(group['seed_status_counts'].items()))
        return '— (' + counts + ')'
    return f"{value['mean']:.6g} ± {value['std_population']:.3g}"


def markdown(report):
    abc, ref = report['abc'], report['tflow_reference']
    lines = ['# RAFM A/B/C completion report', '',
        f"Snapshot generated **{report['generated_at_paris']} (Paris)**; results collected at {report['source_results_at_utc']}.", '',
        f"**{abc['status']}**: {abc['complete_runs']}/252 verified complete, {abc['failed_runs']} failed, "
        f"{abc['missing_runs']} missing, {abc['in_progress_or_interrupted_runs']} in progress or interrupted, "
        f"{abc['awaiting_sample_audit_runs']} awaiting sample audit, {abc['other_unresolved_runs']} otherwise unresolved.", '',
        f"{len(report['complete_abc_conditions'])}/28 conditions have all nine A/B/C seeds verified. "
        'Scientific failures are recorded outcomes, not missing jobs or zero measurements. '
        'Finished-with-failures means every prescribed outcome was recorded; it does not mean all experiments passed.', '',
        report['scheduler_note'], '',
        'A = original RAFM-Ang; B = unit direction plus radius; C = unit direction with constant-zero radius input. '
        'Every displayed aggregate requires three compatible audited seeds and uses population SD.', '',
        '| Condition | Primary metric | A | B | C | Existing t-Flow reference |',
        '|---|---|---|---|---|---|']
    for condition in report['conditions']:
        metric = findings.PRIMARY[condition['kind']][0]
        line = [condition['condition_id'], metric] + [cell(condition['methods'][arm], metric) for arm in ARMS]
        line.append(cell(condition['tflow_reference'], metric))
        lines.append('| ' + ' | '.join(line) + ' |')
    lines += ['', '## Measured findings', '']
    for contrast, tally in report['findings'].items():
        outcomes = tally['mean_outcomes']; consistency = tally['seed_consistency']
        lines.append(f"- {contrast}: {tally['eligible_conditions']} complete paired conditions; "
            f"B has a better primary-metric mean in {outcomes['improved']}, a worse mean in {outcomes['worsened']}, "
            f"and ties in {outcomes['tied']}. Better on every seed in {consistency['improved_all_three']}; "
            f"worse on every seed in {consistency['worsened_all_three']}. These are condition counts, not pooled effect sizes.")
    lines += ['', '## Costs and protocol', '',
        '| Condition | Method | Parameters | Training seconds | Sampling seconds |', '|---|---|---|---|---|']
    for condition in report['conditions']:
        for method, group in condition['methods'].items():
            params = group.get('parameter_summary', {}).get('total_parameters')
            value = str(int(params['mean'])) if group['status'] == 'complete' and params else '— (see per-seed records)'
            lines.append('| ' + ' | '.join([condition['condition_id'], method, value,
                cell(group, 'total_train_time_s'), cell(group, 'sample_time_s')]) + ' |')
    lines += ['', 'Per-seed parameter overhead, peak memory, hardware, precision, actual network calls, '
        'radius drift and timing scopes are retained in abc_completion.json. '
        'Matched runtime ratios are included only where the recorded environments match.', '',
        '## Existing t-Flow reference', '',
        f"{ref['complete_runs']}/84 complete, {ref['failed_runs']} failed, {ref['missing_runs']} missing; "
        'these counts are separate from the 252 A/B/C runs. Missing recovered-condition t-Flow jobs are not failures '
        'and do not block A/B/C completion. No tuning or training is performed by this reporter.', '',
        'The qualified comparison concerns this published-noise-objective adaptation to matched backbones. '
        'Undefined metrics and nonfinite samples remain failures; finite metrics from failed runs are retained only per seed.', '',
        '## All per-seed outcomes', '', '| Condition | Method | Seed | Status | Primary value | Record |',
        '|---|---|---|---|---|---|']
    for condition in report['conditions']:
        metric = findings.PRIMARY[condition['kind']][0]
        for method, group in list(condition['methods'].items()) + [('tflow_reference', condition['tflow_reference'])]:
            for row in group['seeds']:
                value = row['metrics'].get(metric)
                text = f'{value:.8g}' if shared.numeric(value) else '—'
                if row['status'] != 'complete' and shared.numeric(value):
                    text += ' (partial; excluded from means)'
                path = row['result_path']
                link = f'[result.json](<{path}>)' if row.get('result_sha256') else f'`{path}` (not present)'
                lines.append('| ' + ' | '.join([condition['condition_id'], method, str(row['seed']), row['status'], text, link]) + ' |')
    lines += ['', '## Failures and unresolved records', '']
    failures = 0
    for condition in report['conditions']:
        for method, group in list(condition['methods'].items()) + [('tflow_reference', condition['tflow_reference'])]:
            for row in group['seeds']:
                if row['status'] in ('complete', 'missing', 'in_progress_or_interrupted', 'awaiting_sample_audit'):
                    continue
                failures += 1
                failure = row.get('failure') or {}
                reason = '; '.join(row.get('issues', [])) or failure.get('message') or ', '.join(row.get('nonfinite_metric_fields') or []) or row['status']
                lines.append(f"- {condition['condition_id']} / {method} / seed {row['seed']}: {row['status']}; {reason}.")
    if not failures:
        lines.append('No failure or incompatibility records in this snapshot.')
    lines += ['', '## Interpretation limits', ''] + ['- ' + text for text in report['limitations']]
    for document in report['failure_analysis_documents']:
        lines.append('- [Failure analysis: ' + Path(document['path']).name + '](<' + document['path'] + '>).')
    for condition in report['conditions']:
        notes = condition['provenance_notes']; overlap = notes['image_reference_overlap']
        if overlap:
            counts = overlap.get('generator_split_counts', {})
            lines.append(f"- Image reference protocol is preserved as authorized: {overlap.get('n_reference')} PNGs, "
                f"overlap with generator train/validation/test = {counts.get('train')}/{counts.get('val')}/{counts.get('test')}. "
                'This is not a disjoint held-out image reference. No reference regeneration or protocol correction was applied.')
        if notes['recovered_commit_caveat']:
            lines.append('- ' + condition['condition_id'] + ': ' + notes['recovered_commit_caveat'])
    lines += ['', 'The fixed-spherical + empirical-gain AudioMNIST result remains a separately verified checkpoint reference '
        'in the JSON/audio artifacts. It does not replace arm A or the RAFM-Ang versus RAFM-Vel ablation.', '']
    return '\n'.join(lines)


def write_outputs(report, output):
    output.mkdir(parents=True, exist_ok=True)
    shared.write_json(output / FILES[0], report)
    atomic_text(output / FILES[1], markdown(report))
    statuses, metrics, aggregates, references = [], [], [], []
    for condition in report['conditions']:
        name, label = condition['condition_id'], condition['comparison_label']
        for method, group in list(condition['methods'].items()) + [('tflow', condition['tflow_reference'])]:
            for row in group['seeds']:
                if method in ARMS:
                    statuses.append([name, label, method, row['seed'], row['status'], row['result_path'],
                        row.get('result_sha256', ''), json.dumps(row.get('issues', [])),
                        json.dumps(row.get('failure')), json.dumps(row.get('nonfinite_metric_fields') or [])])
                target = references if method == 'tflow' else metrics
                for key, value in sorted(row['metrics'].items()) or [('', '')]:
                    target.append([name, label, method, row['seed'], row['status'], key,
                        json.dumps(value, allow_nan=False), row['result_path']])
            if method in ARMS:
                for key, value in sorted(group['aggregate'].items()) or [('', {})]:
                    aggregates.append([name, label, method, group['status'], key, value.get('mean', ''),
                        value.get('std_population', ''), value.get('n', ''), json.dumps(value.get('seeds', []))])
    atomic_text(output / FILES[2], csv_text(['condition', 'comparison_label', 'method', 'seed', 'status',
        'result_path', 'result_sha256', 'issues', 'failure', 'nonfinite_metric_fields'], statuses))
    header = ['condition', 'comparison_label', 'method', 'seed', 'status', 'metric', 'value', 'result_path']
    atomic_text(output / FILES[3], csv_text(header, metrics))
    atomic_text(output / FILES[4], csv_text(['condition', 'comparison_label', 'method', 'status', 'metric',
        'mean', 'std_population', 'n', 'seeds'], aggregates))
    atomic_text(output / FILES[5], csv_text(header, references))
    shared.write_json(output / 'abc_completion_files.json', {'generated_at_paris': report['generated_at_paris'],
        'files': [{'path': str(output / name), 'sha256': shared.sha256(output / name)} for name in FILES]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-root', type=Path, default=CONFIG_ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--report', type=Path, help='Consume an existing validated 28-condition report snapshot instead of collecting current files')
    parser.add_argument('--require-finished', action='store_true', help='Exit 2 after writing if prescribed A/B/C outcomes are still unresolved; scientific failures remain explicit finished outcomes')
    args = parser.parse_args()
    if args.report:
        source = shared.read_json(args.report)
        record = {'mode': 'existing_snapshot', 'path': str(args.report.resolve()), 'sha256': shared.sha256(args.report)}
    else:
        source = shared.collect(args.config_root, ROOT / 'outputs_rafm_input_study/v1/final',
            ROOT / 'outputs_tflow_full/v2/final',
            ROOT / 'outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/aggregate.json',
            ROOT / 'outputs_audio_gain/delivery_manifest.json')
        record = {'mode': 'fresh_read_only_collection', 'content_sha256': shared.config_hash(source),
                  'collector': str(ROOT / 'tools/report_shared_study.py')}
    report = build_report(source, args.config_root, record)
    write_outputs(report, args.output.resolve())
    print(json.dumps({'status': report['status'], 'abc': report['abc'],
        'tflow_reference': report['tflow_reference'], 'paris_time': report['generated_at_paris'],
        'report': str((args.output / 'abc_completion.md').resolve())}, indent=2, allow_nan=False))
    if args.require_finished and report['status'] == 'incomplete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
