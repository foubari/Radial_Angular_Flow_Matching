"""Measured findings for the complete RAFM input/t-Flow report; stdlib only.

Answers are scoped to complete compatible three-seed comparisons. No synthetic
performance values, survivor averages, pooled metric units or significance claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = {'vector': ('sliced_w1', 'lower'), 'audio': ('digit_acc', 'higher'), 'image': ('fid', 'lower')}
ANGULAR = ['angular_sw_mean'] + [f'angular_sw_bin{i}' for i in range(4)]


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values):
    if len(values) != 3 or not all(numeric(value) for value in values):
        raise ValueError('Findings require three finite seed values')
    return {'mean': statistics.mean(values), 'std_population': statistics.pstdev(values), 'values': values, 'n': 3}


def complete_rows(group, seeds):
    if len(seeds) != 3 or len(set(seeds)) != 3 or group.get('status') != 'complete':
        return None
    rows = group.get('seeds', [])
    if len(rows) != 3 or {row.get('seed') for row in rows} != set(seeds):
        return None
    if any(row.get('status') != 'complete' for row in rows):
        return None
    for key in ('config_sha256', 'dataset_manifest'):
        if not rows[0].get(key) or any(row.get(key) != rows[0][key] for row in rows[1:]):
            return None
    mapping = {row['seed']: row for row in rows}
    return [mapping[seed] for seed in seeds]


def matched_rows(condition, left, right):
    seeds = condition['seeds']
    first = complete_rows(condition['methods'][left], seeds)
    second = complete_rows(condition['methods'][right], seeds)
    if first is None or second is None:
        return None, 'both methods need complete three-seed groups'
    if condition.get('blocking_issues'):
        return None, 'condition still has unresolved protocol blockers'
    for a, b in zip(first, second):
        if not a.get('config_sha256') or a['config_sha256'] != b.get('config_sha256') or a['config_sha256'] != condition.get('config_sha256'):
            return None, 'paired configuration identity is missing or different'
        if not a.get('dataset_manifest') or a['dataset_manifest'] != b.get('dataset_manifest'):
            return None, 'paired realized dataset/split identity is missing or different'
    if {left, right} == {'B', 'C'} and condition.get('B_C_parameter_equality') != 'passed':
        return None, 'B/C parameter-count equality is not verified'
    return (first, second), None


def metric_effect(first, second, seeds, metric, direction):
    a = [row.get('metrics', {}).get(metric) for row in first]
    b = [row.get('metrics', {}).get(metric) for row in second]
    if not all(numeric(value) for value in a + b):
        return {'status': 'unavailable', 'reason': 'metric missing or nonfinite for at least one paired seed'}
    deltas = [x - y for x, y in zip(a, b)]
    signed = deltas if direction == 'higher' else [-value for value in deltas]
    average = statistics.mean(deltas)
    signed_average = average if direction == 'higher' else -average
    labels = ['improved' if value > 0 else 'worsened' if value < 0 else 'tied' for value in signed]
    counts = dict(Counter(labels))
    return {'status': 'measured', 'metric': metric, 'better_direction': direction,
            'left': stats(a), 'right': stats(b), 'delta_left_minus_right': stats(deltas),
            'seeds': seeds, 'per_seed': [{'seed': seed, 'left': x, 'right': y, 'delta': delta, 'outcome': label}
                                       for seed, x, y, delta, label in zip(seeds, a, b, deltas, labels)],
            'mean_outcome': 'improved' if signed_average > 0 else 'worsened' if signed_average < 0 else 'tied',
            'seed_outcome_counts': counts,
            'consistency': labels[0] + '_all_three' if len(set(labels)) == 1 else 'mixed_or_tied_seeds'}


def runtime_ratios(first, second, condition):
    required = ('gpu', 'torch', 'python', 'cuda', 'hip')
    precision = condition.get('training', {}).get('precision')
    issues = []
    if not precision:
        issues.append('training precision is not recorded')
    for a, b in zip(first, second):
        ha, hb = a.get('hardware', {}), b.get('hardware', {})
        absent = [key for key in required if key not in ha or key not in hb]
        different = [key for key in required if key in ha and key in hb and ha[key] != hb[key]]
        if absent or different:
            issues.append(f"seed {a.get('seed')}: missing hardware fields {absent}; different fields {different}")
    if issues:
        return {'status': 'unresolved_hardware_or_precision', 'issues': issues, 'ratios': {}}
    ratios = {}
    for metric in ('total_train_time_s', 'sample_time_s'):
        a = [row.get('metrics', {}).get(metric) for row in first]
        b = [row.get('metrics', {}).get(metric) for row in second]
        if not all(numeric(value) and value > 0 for value in a + b):
            ratios[metric] = {'status': 'unavailable', 'reason': 'positive measured time missing for one or more seeds'}
            continue
        ratios[metric] = {'status': 'measured', 'per_seed_ratio_left_over_right': stats([x / y for x, y in zip(a, b)]),
                          'ratio_of_mean_seconds': statistics.mean(a) / statistics.mean(b),
                          'left_mean_seconds': statistics.mean(a), 'right_mean_seconds': statistics.mean(b)}
    return {'status': 'recorded_hardware_and_precision_match', 'precision': precision, 'ratios': ratios,
            'note': 'Raw measured ratios, not a significance claim or throughput guarantee. Recorded loops include their compile/checkpoint/instrumentation scopes; no historical timing is used.'}


def comparison(condition, left, right):
    rows, reason = matched_rows(condition, left, right)
    result = {'condition_id': condition['condition_id'], 'kind': condition['kind'], 'contrast': left + '-' + right,
              'comparison_label': condition.get('comparison_label'), 'status': 'unavailable', 'reason': reason,
              'primary': {}, 'angular': {}, 'runtime': {}}
    if rows is None:
        return result
    first, second = rows
    metric, direction = PRIMARY[condition['kind']]
    result['primary'] = metric_effect(first, second, condition['seeds'], metric, direction)
    result['angular'] = {key: metric_effect(first, second, condition['seeds'], key, 'lower') for key in ANGULAR}
    result['runtime'] = runtime_ratios(first, second, condition)
    result['status'] = 'measured' if result['primary']['status'] == 'measured' else 'unavailable'
    return result


def parameter_cost(condition):
    result = {}
    for method, group in condition['methods'].items():
        rows = complete_rows(group, condition['seeds'])
        if rows is None:
            result[method] = {'status': 'unavailable'}
            continue
        selected = ('total_parameters', 'original_backbone_parameters', 'conditioning_overhead_parameters', 'conditioning_overhead_fraction')
        fields = {}
        issues = []
        for key in selected:
            values = [row.get('parameters', {}).get(key) for row in rows]
            if all(numeric(value) for value in values) and len(set(values)) == 1:
                fields[key] = values[0]
            else:
                issues.append(key + ' is absent or differs across seeds')
        result[method] = {'status': 'recorded' if 'total_parameters' in fields else 'unavailable', 'fields': fields, 'unresolved': issues}
    return result


def outcome_tally(comparisons, field='primary'):
    rows = [item for item in comparisons if item.get('status') == 'measured' and item[field].get('status') == 'measured']
    count = Counter(item[field]['mean_outcome'] for item in rows)
    consistency = Counter(item[field]['consistency'] for item in rows)
    return {'eligible_conditions': len(rows), 'mean_outcomes': {key: count[key] for key in ('improved', 'worsened', 'tied')},
            'seed_consistency': {key: consistency[key] for key in ('improved_all_three', 'worsened_all_three', 'tied_all_three', 'mixed_or_tied_seeds')},
            'conditions': [item['condition_id'] for item in rows],
            'interpretation': 'Counts of condition-specific outcomes, not an average of metric units or independent statistical replicates.'}


def rankings(condition):
    metric, direction = PRIMARY[condition['kind']]
    available = {}
    for method, group in condition['methods'].items():
        rows = complete_rows(group, condition['seeds'])
        if rows is None or condition.get('blocking_issues') or any(row.get('config_sha256') != condition.get('config_sha256') for row in rows):
            continue
        values = [row.get('metrics', {}).get(metric) for row in rows]
        if all(numeric(value) for value in values):
            available[method] = stats(values)
    ordered = sorted(available, key=lambda name: available[name]['mean'], reverse=direction == 'higher')
    return {'metric': metric, 'better_direction': direction, 'methods': available,
            'order_by_measured_mean': ordered, 'best_mean_methods': [name for name in ordered if available[name]['mean'] == available[ordered[0]]['mean']] if ordered else [],
            'missing_methods': [name for name in condition['methods'] if name not in available],
            'scope': 'Ranking only among listed complete three-seed methods; close means are not statistically separated.'}


def tuning_cost(condition):
    records = condition.get('tflow_tuning', {}).get('records', [])
    candidates = [item['record'] for item in records if isinstance(item.get('record'), dict)
                  and item['record'].get('status') == 'frozen' and 'stage1_trials' in item['record']]
    if len(candidates) != 1:
        return {'status': 'unavailable', 'reason': 'exactly one frozen complete selection record is required'}
    selection = candidates[0]
    if selection.get('selection_split') != 'validation' or selection.get('test_data_used_for_selection') is not False:
        return {'status': 'unavailable', 'reason': 'validation-only selection is not verified'}
    if selection.get('config_sha256') != condition.get('config_sha256'):
        return {'status': 'unavailable', 'reason': 'selection configuration identity differs'}
    stage1, finalists = selection.get('stage1_trials', []), selection.get('finalist_trials', [])
    by_index = {row.get('candidate_index'): row for row in stage1}
    if len(stage1) != 9 or set(by_index) != set(range(9)) or len(finalists) != 2 or len({row.get('candidate_index') for row in finalists}) != 2:
        return {'status': 'unavailable', 'reason': 'expected nine initial trials and two distinct continuations'}
    steps = condition.get('training', {}).get('steps')
    increments = []
    for row in stage1 + finalists:
        if row.get('status') != 'complete' or not all(numeric(row.get(key)) and row[key] >= 0 for key in ('training_time_s', 'sample_time_s')):
            return {'status': 'unavailable', 'reason': 'complete nonnegative trial timings are missing'}
        if row.get('config_sha256') != selection['config_sha256'] or row.get('implementation_sha256') != selection.get('implementation_sha256'):
            return {'status': 'unavailable', 'reason': 'trial implementation/configuration identity differs'}
    if not numeric(steps) or any(row.get('steps') != int(steps * .05) for row in stage1):
        return {'status': 'unavailable', 'reason': 'initial trial update budgets differ from the documented 5%'}
    for row in finalists:
        original = by_index.get(row.get('candidate_index'))
        if original is None or row.get('source') != original.get('source') or row.get('steps') != int(steps * .10):
            return {'status': 'unavailable', 'reason': 'continued finalist identity or 10% update budget is unresolved'}
        delta = row['training_time_s'] - original['training_time_s']
        if delta < 0:
            return {'status': 'unavailable', 'reason': 'finalist cumulative time is shorter than its initial trial; no guessed continuation cost'}
        increments.append({'candidate_index': row['candidate_index'], 'additional_training_loop_s': delta})
    initial = sum(row['training_time_s'] for row in stage1)
    continuation = sum(row['additional_training_loop_s'] for row in increments)
    sampling = sum(row['sample_time_s'] for row in stage1 + finalists)
    return {'status': 'measured_from_complete_trial_records', 'stage1_training_loop_s': initial,
            'continuation_training_loop_s': continuation, 'total_training_loop_s': initial + continuation,
            'validation_sampling_s': sampling, 'continuations': increments,
            'training_updates': 9 * int(steps * .05) + 2 * (int(steps * .10) - int(steps * .05)),
            'full_training_equivalents': .55, 'selected_source': selection.get('selected'),
            'scope': 'Sum initial cumulative loop times plus each finalist cumulative time minus its own initial time. Sampling includes all nine initial and two continuation evaluations. Selection metrics, data loading and scheduler overhead are not timed here.'}


def audio_reference_comparison(report):
    ref = report.get('fixed_spherical_gain_reference', {})
    audio = next((condition for condition in report['conditions'] if condition['kind'] == 'audio'), None)
    result = {'status': 'unavailable', 'reason': 'needs verified reference and completed compatible new audio methods',
              'historical_reproduction_limitation': ref.get('reproduction_limitation', 'Historical accuracy reproduction remains unresolved.'),
              'historical_fixed_accuracy': ref.get('historical_fixed_accuracy'),
              'methods': {}, 'not_a_training_runtime_comparison': True, 'backend_compatibility': ref.get('backend_compatibility', {})}
    if audio is None or not ref.get('eligible_for_prepared_protocol') or ref.get('training_seeds') != audio['seeds']:
        return result
    if ref.get('backend_compatibility', {}).get('verified') is not True:
        result['reason'] = 'saved-output evaluation under the new backend is not verified; zero prediction changes and matching energy/tails must be measured'
        return result
    checks = ref.get('new_audio_execution_compatibility', [])
    for method, group in audio['methods'].items():
        rows = complete_rows(group, audio['seeds'])
        if rows is None:
            continue
        matching = [item for item in checks if item.get('method') == method and item.get('status') == 'verified']
        if {item.get('seed') for item in matching} != set(audio['seeds']) or len(matching) != 3:
            continue
        metrics = {}
        for key in ('digit_acc', 'energy_KS', 'cov>q95', 'cov>q99'):
            values = [row.get('metrics', {}).get(key) for row in rows]
            if all(numeric(value) for value in values):
                metrics[key] = stats(values)
        if len(metrics) == 4:
            result['methods'][method] = metrics
    if not result['methods']:
        return result
    reference_metrics = {}
    for key in ('digit_acc', 'energy_KS', 'cov>q95', 'cov>q99'):
        values = ref.get('measured_full_precision', {}).get(key, {}).get('vals', [])
        if len(values) != 3 or not all(numeric(value) for value in values):
            return dict(result, status='unavailable', reason='complete measured reference values missing')
        reference_metrics[key] = stats(values)
    result['methods']['fixed_spherical_empirical_gain_reference'] = reference_metrics
    best = max(values['digit_acc']['mean'] for values in result['methods'].values())
    result.update({'status': 'measured_compatible_comparison', 'reason': None,
                  'best_digit_accuracy_methods': [method for method, values in result['methods'].items() if values['digit_acc']['mean'] == best],
                  'scope': 'Best measured mean only among listed complete compatible new audio methods and the checkpoint-based reference. This is not a significance claim; gain is constructed independently of content.'})
    result['new_method_accuracy_minus_reference'] = {method: values['digit_acc']['mean'] - reference_metrics['digit_acc']['mean']
                                                    for method, values in result['methods'].items() if method != 'fixed_spherical_empirical_gain_reference'}
    return result


def build_findings(report):
    result = {'schema_version': 1, 'study_id': report.get('study_id'), 'report_generated_at_utc': report.get('generated_at_utc'),
              'status': 'complete_suite' if report.get('status') == 'complete' else 'partial_suite',
              'coverage': {'expected_conditions': len(report['conditions']), 'expected_final_runs': report.get('expected_final_runs'),
                           'seed_status_counts': report.get('seed_status_counts', {}),
                           'complete_conditions': report.get('complete_conditions', []),
                           'blocked_conditions': [{'condition': condition['condition_id'], 'issues': condition['blocking_issues']}
                                                  for condition in report['conditions'] if condition.get('blocking_issues')]},
              'comparisons': {'B-A': [], 'B-C': [], 'tflow-A': []}, 'parameters': {}, 'tflow_rankings': {}, 'tflow_tuning': {},
              'limitations': [
                  'New synthetic realizations are shared A/B/C/t-Flow comparisons, never historical tensors or replacement historical results.',
                  'Three-seed mean differences and sign consistency are descriptive, not tests of statistical significance.',
                  'No pooling of metric units across datasets, no averaging over missing/failed seeds, and no data-dependent tolerance for a tie.',
                  'B versus A changes input parameterization and includes radius conditioning; B versus C isolates access to the scalar radius in identical modules.',
                  'Condition sweeps share related generators and are not independent statistical replicates.',
                  'This input reparameterization preserves the population angular transport objective and creates no new theoretical guarantee.']}
    for condition in report['conditions']:
        for left, right in [('B', 'A'), ('B', 'C'), ('tflow', 'A')]:
            result['comparisons'][left + '-' + right].append(comparison(condition, left, right))
        result['parameters'][condition['condition_id']] = parameter_cost(condition)
        result['tflow_rankings'][condition['condition_id']] = rankings(condition)
        result['tflow_tuning'][condition['condition_id']] = tuning_cost(condition)
    result['tallies'] = {name: outcome_tally(rows) for name, rows in result['comparisons'].items()}
    result['tallies_by_kind'] = {name: {kind: outcome_tally([item for item in rows if item['kind'] == kind]) for kind in PRIMARY}
                               for name, rows in result['comparisons'].items()}
    result['angular_tallies'] = {}
    for name, rows in result['comparisons'].items():
        result['angular_tallies'][name] = {}
        for metric in ANGULAR:
            projected = [dict(item, primary=item['angular'].get(metric, {})) for item in rows]
            result['angular_tallies'][name][metric] = outcome_tally(projected)
    result['audio_reference'] = audio_reference_comparison(report)
    operations = report.get('operations', {})
    result['separate_failures'] = {
        'implementation_check_failures': operations.get('implementation_check_failures', []),
        'previous_tflow_attempts': operations.get('previous_tflow_attempts', []),
        'interpretation': 'Preserved preparation/startup failures remain separate from current final scientific seed status counts.'}
    answers = {}
    for question, name, meaning in [(1, 'B-A', 'Normalized input plus radius conditioning (B) versus original RAFM-Ang (A)'),
                                    (2, 'B-C', 'Explicit radius conditioning (B) versus identical zero-conditioned modules (C)')]:
        tally = result['tallies'][name]
        n = tally['eligible_conditions']
        if n:
            count = tally['mean_outcomes']
            answers[str(question)] = (f"{meaning}: among {n}/{len(report['conditions'])} eligible conditions, the primary metric improves in {count['improved']}, worsens in {count['worsened']}, and ties in {count['tied']}. "
                                      f"All three seeds favor B in {tally['seed_consistency']['improved_all_three']} conditions. These are condition-specific descriptive outcomes; individual effects follow below.")
        else:
            answers[str(question)] = meaning + ': no complete compatible three-seed pair is available yet; no measured effect can be assigned.'
    pieces = []
    for name in ('B-A', 'B-C'):
        tally = result['tallies'][name]
        c = tally['seed_consistency']
        pieces.append(f"{name}: {c['improved_all_three']} conditions improve in every seed, {c['worsened_all_three']} worsen in every seed, {c['tied_all_three']} tie in every seed, and {c['mixed_or_tied_seeds']} have mixed/tied seed outcomes")
    answers['3'] = '; '.join(pieces) + '. Coverage and individual deltas, including negative results, are reported without a cross-dataset pooled score.'
    measured_cost = {name: sum(item.get('runtime', {}).get('status') == 'recorded_hardware_and_precision_match' for item in result['comparisons'][name]) for name in ('B-A', 'B-C')}
    answers['4'] = (f"Recorded hardware/precision permit descriptive timing ratios for {measured_cost['B-A']} B/A and {measured_cost['B-C']} B/C condition pairs. "
                    'The parameter table reports absolute counts and the recorded conditioning overhead; B/C receive no additional tuning. t-Flow validation-only training and sampling seconds are separately reconstructed only from complete, consistent trial records.')
    result['current_final_failures'] = [{'condition': condition['condition_id'], 'method': method, 'seed': row.get('seed'),
        'status': row.get('status'), 'failure': row.get('failure'), 'issues': row.get('issues'), 'result_path': row.get('result_path')}
        for condition in report['conditions'] for method, group in condition['methods'].items()
        for row in group.get('seeds', []) if row.get('status') in ('failed', 'incompatible', 'unreadable', 'unrecognized_status')]
    result['current_tuning_failures'] = [{'condition': condition['condition_id'], 'path': entry.get('path'), 'record': entry['record']}
        for condition in report['conditions'] for entry in condition.get('tflow_tuning', {}).get('records', [])
        if isinstance(entry.get('record'), dict) and entry['record'].get('status') == 'failed']
    result['answers'] = answers
    return result


def number(value):
    return f'{value:.7g}' if numeric(value) else 'unavailable'


def write_findings(result, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix='.findings-', dir=directory)
    with os.fdopen(fd, 'w') as stream:
        stream.write(payload)
    os.replace(temporary, directory / 'findings.json')
    coverage = result['coverage']
    lines = ['# Measured study findings', '', f"Scope: **{result['status']}**; {len(coverage['complete_conditions'])}/{coverage['expected_conditions']} conditions have all A/B/C/t-Flow methods complete.", '',
             'Final seed status counts: ' + json.dumps(coverage['seed_status_counts'], sort_keys=True) + '.', '',
             'New synthetic realizations are new shared comparisons; published historical results remain unchanged.', '']
    for question, text in result['answers'].items():
        lines += [f'**{question}.** {text}', '']
    lines += ['## Paired primary effects', '', '| Condition | Contrast | Metric (better) | Left mean ± SD | Right mean ± SD | Mean delta | Three seed deltas | Mean outcome / consistency |', '|---|---|---|---|---|---|---|---|']
    for name, comparisons in result['comparisons'].items():
        for item in comparisons:
            if item['status'] != 'measured':
                continue
            p = item['primary']; a, b = p['left'], p['right']
            cells = [item['condition_id'], name, p['metric'] + ' (' + p['better_direction'] + ')',
                     number(a['mean']) + ' ± ' + number(a['std_population']), number(b['mean']) + ' ± ' + number(b['std_population']),
                     number(p['delta_left_minus_right']['mean']), ', '.join(number(value) for value in p['delta_left_minus_right']['values']), p['mean_outcome'] + ' / ' + p['consistency']]
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Deltas are left minus right. Positive favors the left method for digit accuracy; negative favors it for sliced W1/FID. Standard deviations are population SD. Seed order is recorded per condition in findings.json.', '',
              '## Conditional angular effects', '', '| Condition | Contrast | Angular metric | Mean delta | Three seed deltas | Mean outcome / consistency |', '|---|---|---|---|---|---|']
    for name, comparisons in result['comparisons'].items():
        for item in comparisons:
            for key, metric in item['angular'].items():
                if metric.get('status') != 'measured':
                    continue
                lines.append('| ' + ' | '.join([item['condition_id'], name, key, number(metric['delta_left_minus_right']['mean']),
                                                ', '.join(number(value) for value in metric['delta_left_minus_right']['values']),
                                                metric['mean_outcome'] + ' / ' + metric['consistency']]) + ' |')
    lines += ['', 'Angular sliced W1 is lower-is-better. Empty/nonfinite bins and incomplete seed groups do not acquire averages.', '',
              '## Recorded cost', '', '| Condition | Ratio | Training: mean seed ratio | Sampling: mean seed ratio |', '|---|---|---|---|']
    for name in ('B-A', 'B-C'):
        for item in result['comparisons'][name]:
            runtime = item.get('runtime', {})
            if runtime.get('status') != 'recorded_hardware_and_precision_match':
                continue
            cells = [item['condition_id'], name.replace('-', '/')]
            for key in ('total_train_time_s', 'sample_time_s'):
                ratio = runtime['ratios'][key]
                cells.append(number(ratio['per_seed_ratio_left_over_right']['mean']) if ratio['status'] == 'measured' else 'unavailable')
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'These are raw ratios of measured times on matching recorded GPU/software and precision, not significance claims. Ratios of means and each seed ratio are also saved in findings.json.', '',
              '| Condition | Method | Total parameters | Added conditioning parameters | Added fraction |', '|---|---|---|---|---|']
    for condition, methods in result['parameters'].items():
        for method, record in methods.items():
            if record['status'] != 'recorded':
                continue
            fields = record['fields']
            lines.append('| ' + ' | '.join([condition, method] + [number(fields.get(key)) for key in ('total_parameters', 'conditioning_overhead_parameters', 'conditioning_overhead_fraction')]) + ' |')
    lines += ['', '## t-Flow', '']
    tally = result['tallies']['tflow-A']
    lines += [f"Against newly run A, t-Flow has {tally['eligible_conditions']} complete comparisons: " + json.dumps(tally['mean_outcomes'], sort_keys=True) + ' on each condition’s primary metric. Ranking is descriptive and limited to the complete methods listed below.', '',
              '| Condition | Primary metric | Methods ordered by mean | Missing methods |', '|---|---|---|---|']
    for condition, rank in result['tflow_rankings'].items():
        if 'tflow' in rank['methods'] and len(rank['methods']) >= 2:
            lines.append('| ' + ' | '.join([condition, rank['metric'], ', '.join(rank['order_by_measured_mean']), ', '.join(rank['missing_methods']) or 'none']) + ' |')
    lines += ['', '| Condition | Validation training-loop seconds | Validation sampling seconds | Training-update equivalents |', '|---|---|---|---|']
    for condition, timing in result['tflow_tuning'].items():
        if timing['status'] == 'measured_from_complete_trial_records':
            lines.append('| ' + ' | '.join([condition, number(timing['total_training_loop_s']), number(timing['validation_sampling_s']), number(timing['full_training_equivalents'])]) + ' |')
    lines += ['', 'Tuning training time sums initial cumulative times plus each finalist’s cumulative time minus its own initial time. It does not double-count continuation checkpoints. Validation metrics, data loading and scheduling overhead are unmeasured here; incomplete/inconsistent trials are marked unavailable.', '',
              '## Audio checkpoint reference', '']
    audio = result['audio_reference']
    if audio['status'] == 'measured_compatible_comparison':
        lines += ['Best measured digit-accuracy mean among the complete compatible listed methods: **' + ', '.join(audio['best_digit_accuracy_methods']) + '**.', '',
                  '| Method | Accuracy | Energy KS | Coverage > q95 | Coverage > q99 |', '|---|---|---|---|---|']
        for method, metrics in audio['methods'].items():
            lines.append('| ' + ' | '.join([method] + [number(metrics[key]['mean']) + ' ± ' + number(metrics[key]['std_population']) for key in ('digit_acc', 'energy_KS', 'cov>q95', 'cov>q99')]) + ' |')
    else:
        lines += [audio['reason'] + '.']
    lines += ['', audio['historical_reproduction_limitation'], '',
              'The checkpoint-based fixed-spherical+gain row is not new A and is not used for matched training-time claims. Constructed gains are independent of content; a B/C benefit is not presumed. The RAFM-Ang versus RAFM-Vel interpretation remains separate.', '',
              '## Missing, failed and blocked work', '']
    for item in coverage['blocked_conditions']:
        lines.append('- ' + item['condition'] + ': ' + ', '.join(item['issues']))
    for failure in result.get('current_final_failures', []):
        lines.append('- Current final seed ' + failure['condition'] + ' / ' + failure['method'] + ' / ' + str(failure['seed']) + ': ' + failure['status'] + '; ' + json.dumps(failure.get('failure') or failure.get('issues'), sort_keys=True))
    for failure in result.get('current_tuning_failures', []):
        lines.append('- Current t-Flow tuning failure ' + failure['condition'] + ': ' + str(failure['record'].get('error', failure['record'].get('failure'))) + '; ' + str(failure.get('path')))
    for name, comparisons in result['comparisons'].items():
        unavailable = [item['condition_id'] for item in comparisons if item['status'] != 'measured']
        if unavailable:
            lines.append('- Unavailable ' + name + ' comparisons: ' + ', '.join(unavailable))
    separate = result['separate_failures']
    lines += ['', f"Separately preserved: {len(separate['implementation_check_failures'])} implementation-check failure archives and {len(separate['previous_tflow_attempts'])} earlier t-Flow startup-attempt archives. They are not current scientific seed failures.", '',
              '## Interpretation limits', ''] + ['- ' + note for note in result['limitations']]
    (directory / 'findings.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs_rafm_input_study/v1/report/report.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    findings = build_findings(report)
    findings['input_report'] = {'path': str(args.report), 'sha256': hashlib.sha256(raw).hexdigest()}
    output = args.output or args.report.parent
    write_findings(findings, output)
    print(json.dumps({'status': findings['status'], 'eligible_primary_comparisons': {key: value['eligible_conditions'] for key, value in findings['tallies'].items()}, 'findings': str(output / 'findings.md')}, indent=2))


if __name__ == '__main__':
    main()
