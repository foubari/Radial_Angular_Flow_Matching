"""CPU-only plot validation using explicit synthetic report fixtures, not results.

Never loads models or tensor datasets and never writes to final/report outputs.
Every exported figure carries a visible fixture watermark. The one real reference
is the explicitly labelled current-backend reevaluation of saved fixed-spherical
gain outputs, copied unchanged with its recorded historical-backend differences.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CHECK_ROOT = ROOT / 'outputs_rafm_input_study/v1/plot_checks'
REFERENCE_REPORT = ROOT / 'outputs_rafm_input_study/v1/report/report.json'
EXPECTED_FIGURES = {'paired_B_minus_A', 'paired_B_minus_C', 'quality_tflow_and_rafm_inputs',
                    'audio_content_energy_and_tails', 'angular_fit_by_radius_bin',
                    'ambient_sampler_radius_drift', 'measured_runtime_and_parameter_overhead'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seed_fixture(helpers, reporter, kind, method, seed, index, condition):
    cfg, cache, result = helpers.fixture(method if method != 'tflow' else 'A', seed)
    seeds = [8925, 77395, 65457] if kind == 'vector' else [8925, 1234, 7]
    dim, steps, nfe, n_samples = {'vector': (16, 10000, 512, 20),
                                'audio': (16254, 24000, 160, 2000),
                                'image': (2048, 40000, 100, 2000)}[kind]
    cfg.update(condition_id=condition, kind=kind, seeds=seeds)
    cfg['data']['shape'] = [50, dim]
    cfg['training']['steps'] = steps
    cfg['evaluation'].update(n_samples=n_samples, model_evaluations=nfe)
    offset = {'A': 0., 'B': -.02, 'C': .015, 'tflow': .04}[method]
    metrics = {name: .2 + offset + .01 * index for name in reporter.required_metrics(kind, method)}
    metrics.update(nfe=nfe, total_train_time_s=10 + index + {'A': 0, 'B': 2, 'C': 1, 'tflow': -1}[method],
                   sample_time_s=.2 + .01 * index)
    if kind == 'audio':
        metrics.update(digit_acc=.70 - offset + .005 * index, energy_KS=.04 + abs(offset) + .002 * index,
                       **{'cov>q95': .05 + .001 * index, 'cov>q99': .01 + .0005 * index})
    if kind == 'image':
        metrics['fid'] = 12 + 20 * offset + index
    if kind == 'vector' or method != 'tflow':
        metrics.update({name: .08 + abs(offset) + .003 * index for name in reporter.ANGULAR})
    base_count = {'vector': 1000, 'audio': 10000, 'image': 20000}[kind]
    overhead = (16 if kind == 'vector' else 128) if method in ('B', 'C') else 0
    result.update(fixture_only=True, condition_id=condition, seed=seed, config=cfg,
                  config_sha256=reporter.config_hash(cfg), checkpoint={'step': steps}, metrics=metrics,
                  implementation_sha256='fixture-implementation-' + method,
                  sampler={'nfe': nfe, 'n_batches': 1, 'model_calls_total': nfe, 'state_renormalization': False},
                  hardware={'gpu': 'SYNTHETIC FIXTURE HARDWARE', 'torch': 'fixture', 'python': 'fixture',
                            'cuda': None, 'hip': 'fixture'},
                  parameters={'total_parameters': base_count + overhead,
                              'conditioning_overhead_parameters': overhead,
                              'conditioning_overhead_fraction': overhead / base_count},
                  radius_drift={'mean_relative': 1e-6 * (index + 1), 'max_relative': 1e-4 * (index + 1)},
                  peak_memory={'training': {'allocated_bytes': 1000000 + 1000 * index},
                               'sampling': {'allocated_bytes': 2000000 + 1000 * index}})
    if method == 'tflow':
        result.update(method='tflow', stage='final', source={'nu': 5., 'scale': 1.})
        result.pop('arm', None)
        result.pop('radius_drift', None)
        result.pop('parameters', None)
        result['metrics']['n_params'] = base_count
    return cfg, cache, result


def build_fixture(output, reporter, helpers):
    reference_bytes = REFERENCE_REPORT.read_bytes()
    original = json.loads(reference_bytes)
    reference = copy.deepcopy(original['fixed_spherical_gain_reference'])
    if reference.get('status') != 'verified_for_prepared_protocol' or reference.get('eligible_for_prepared_protocol') is not True:
        raise ValueError('The existing fixed-spherical gain reference is not verified')
    current_reference = reference.get('current_backend_reference', {})
    if current_reference.get('usable_for_comparison') is not True:
        raise ValueError('The explicitly labelled measured current-backend reference must be usable and the report must be refreshed before plotting it')
    if sha256(current_reference['audit_path']) != current_reference['audit_sha256']:
        raise ValueError('The current-backend reference audit checksum changed')
    report = {'schema_version': 1, 'fixture_only': True, 'study_id': 'SYNTHETIC_PLOT_FIXTURE_NOT_EXPERIMENT_RESULTS',
              'status': 'fixture_with_complete_and_omitted_groups', 'conditions': [],
              'fixed_spherical_gain_reference': reference,
              'reference_provenance': {'path': str(REFERENCE_REPORT),
                                       'read_bytes_sha256': hashlib.sha256(reference_bytes).hexdigest(),
                                       'reference_section_sha256': reporter.config_hash(reference),
                                       'current_backend_status': current_reference['status'],
                                       'current_backend_audit_path': current_reference['audit_path'],
                                       'current_backend_audit_sha256': current_reference['audit_sha256'],
                                       'archived_backend_compatibility': reference.get('backend_compatibility')}}
    all_seed_rows = []
    for condition, kind in [('fixture_vector_d16', 'vector'), ('audiomnist_stft', 'audio'), ('fixture_image_branch', 'image')]:
        seeds = [8925, 77395, 65457] if kind == 'vector' else [8925, 1234, 7]
        item = {'condition_id': condition, 'kind': kind, 'seeds': seeds, 'fixture_only': True,
                'comparison_label': 'SYNTHETIC REPORT FIXTURE ONLY', 'blocking_issues': [], 'methods': {}}
        for method in reporter.METHODS:
            rows = []
            for index, seed in enumerate(seeds):
                cfg, cache, result = seed_fixture(helpers, reporter, kind, method, seed, index, condition)
                directory = output / 'fixture_seed_records' / condition / method / f'seed_{seed}'
                directory.mkdir(parents=True)
                result_path = directory / 'result.json'
                write(result_path, result)
                audit = {'fixture_only': True, 'status': 'passed', 'result_sha256': sha256(result_path),
                         'sample_sha256': result['sample_artifact']['sha256'], 'n_samples': cfg['evaluation']['n_samples'],
                         'dimension': cfg['data']['shape'][1], 'nfe': cfg['evaluation']['model_evaluations'],
                         'n_batches': 1, 'model_calls_total': cfg['evaluation']['model_evaluations'],
                         'nonfinite_rows': 0, 'nan_rows': 0, 'inf_rows': 0, 'configured_sample_seed': 0,
                         'class_counts': None if kind == 'vector' else [cfg['evaluation']['n_samples'] // 10] * 10,
                         'errors': [], 'note': 'Schema fixture only: no sample tensor exists or was audited.'}
                write(directory / 'sample_audit.json', audit)
                row = reporter.collect_seed(cfg, method, seed, directory, cache)
                if row['status'] != 'complete':
                    raise ValueError(f'Fixture does not satisfy collector schema: {kind}/{method}/{seed}: {row["issues"]}')
                rows.append(row)
            group = reporter.aggregate_group(rows, seeds)
            if group['status'] != 'complete':
                raise ValueError('Fixture triplet aggregation failed: ' + str(group['issues']))
            item['methods'][method] = group
            all_seed_rows.extend(rows)
        item['paired_contrasts'] = {left + '-' + right: reporter.paired(item['methods'], left, right, seeds)
                                   for left, right in [('B', 'A'), ('B', 'C')]}
        if any(contrast['status'] != 'complete' for contrast in item['paired_contrasts'].values()):
            raise ValueError('Fixture paired comparison failed')
        item['B_C_parameter_equality'] = 'passed'
        report['conditions'].append(item)
    for condition, blocked in [('fixture_blocked_condition', True), ('fixture_failed_condition', False)]:
        item = {'condition_id': condition, 'kind': 'vector', 'seeds': [8925, 77395, 65457],
                'blocking_issues': ['fixture missing input, intentionally omitted'] if blocked else [], 'methods': {},
                'paired_contrasts': {name: {'status': 'incomplete', 'metrics': {}} for name in ('B-A', 'B-C')}}
        for method in reporter.METHODS:
            rows = [{'seed': seed, 'status': 'blocked' if blocked else ('failed' if index == 0 else 'missing'), 'metrics': {}}
                    for index, seed in enumerate(item['seeds'])]
            item['methods'][method] = reporter.aggregate_group(rows, item['seeds'])
            all_seed_rows.extend(rows)
        report['conditions'].append(item)
    report['complete_method_groups'] = 12
    report['expected_final_runs'] = len(all_seed_rows)
    report['seed_status_counts'] = dict(Counter(row['status'] for row in all_seed_rows))
    report['audio_comparison_rows'] = reporter.audio_comparison_rows(report)
    if len(report['audio_comparison_rows']) != 5:
        raise ValueError('The fixture must exercise four audio methods and the separate verified gain reference')
    return report


def check(output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Plot fixture validation requires a scheduled CPU allocation')
    output = Path(output).resolve()
    if not output.is_relative_to(CHECK_ROOT.resolve()) or output == CHECK_ROOT.resolve():
        raise ValueError('Write only to a new dedicated child of plot_checks')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    record = {'status': 'running', 'fixture_only': True, 'training_or_benchmark_run': False,
              'gpu_used': False, 'host': platform.node(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'script_sha256': sha256(__file__), 'plot_source_sha256': sha256(ROOT / 'tools/plot_shared_study.py'),
              'collector_source_sha256': sha256(ROOT / 'tools/report_shared_study.py')}
    started = time.perf_counter()
    try:
        reporter = imported('plot_fixture_collector', ROOT / 'tools/report_shared_study.py')
        helpers = imported('plot_fixture_helpers', ROOT / 'tests/test_report_shared_study.py')
        plotter = imported('plot_fixture_plotter', ROOT / 'tools/plot_shared_study.py')
        report = build_fixture(output, reporter, helpers)
        fixture_report = output / 'fixture_report.json'
        write(fixture_report, report)
        (output / 'README.md').write_text('# Synthetic plot checks only\n\nThese are fabricated report fixtures, not experiment measurements. No tensor data or models were loaded. The fixed-spherical gain reference is the one explicitly labelled real reference. Every figure is watermarked. Do not include these files in scientific result aggregation.\n')
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        original_save = Figure.savefig
        saves = []

        def save_fixture(figure, filename, *args, **kwargs):
            if not getattr(figure, '_fixture_watermarked', False):
                figure.text(.5, .5, 'SYNTHETIC FIXTURE\nNOT EXPERIMENT RESULTS', ha='center', va='center',
                            rotation=25, fontsize=28, color='#a33', alpha=.20, zorder=100)
                figure._fixture_watermarked = True
            figure.canvas.draw()
            numeric_ylabels = []
            for axis in figure.axes:
                low, high = sorted(axis.get_ylim())
                visible = [{'position': float(position), 'label': label.get_text()}
                           for position, label in zip(axis.get_yticks(), axis.get_yticklabels())
                           if low <= position <= high and label.get_visible()
                           and any(character.isdigit() for character in label.get_text())]
                if axis.has_data() and not visible:
                    raise AssertionError(f'Data panel has no visible in-range numeric y ticks: {Path(filename).name}, {axis.get_ylabel()}')
                numeric_ylabels.append(visible)
            saves.append({'path': str(filename), 'axes': len(figure.axes), 'watermarked': True,
                          'visible_numeric_ylabels': numeric_ylabels})
            return original_save(figure, filename, *args, **kwargs)

        with patch.object(Figure, 'savefig', new=save_fixture):
            manifest = plotter.plot_report(report, output / 'fixture_figures')
        names = {figure['name'] for figure in manifest['figures']}
        if names != EXPECTED_FIGURES:
            raise AssertionError(f'Expected every plot branch: missing={EXPECTED_FIGURES-names}, extra={names-EXPECTED_FIGURES}')
        if len(manifest['omissions']) != 8:
            raise AssertionError('Blocked and failed fixture groups must remain explicit')
        by_name = {figure['name']: figure for figure in manifest['figures']}
        audio = by_name['audio_content_energy_and_tails']['included']
        if len(audio) != 20 or {row['method'] for row in audio} != set(reporter.METHODS) | {'fixed_spherical_empirical_gain_reference'}:
            raise AssertionError('Audio plot omitted a method/reference/metric')
        quality = by_name['quality_tflow_and_rafm_inputs']['included']
        if len(quality) != 13 or sum(row['method'] == 'fixed_spherical_empirical_gain_reference' for row in quality) != 1:
            raise AssertionError('Quality overview omitted a complete method/domain or the labelled measured reference')
        expected_included = {'paired_B_minus_A': 3, 'paired_B_minus_C': 3,
                             'angular_fit_by_radius_bin': 40, 'ambient_sampler_radius_drift': 18,
                             'measured_runtime_and_parameter_overhead': 33}
        for name, count in expected_included.items():
            if len(by_name[name]['included']) != count:
                raise AssertionError(f'Unexpected plotted item count for {name}: {len(by_name[name]["included"])} != {count}')
        if any(row['method'] == 'tflow' and row['metric'] == 'conditioning_overhead_fraction'
               for row in by_name['measured_runtime_and_parameter_overhead']['included']):
            raise AssertionError('A missing t-Flow conditioning overhead was treated as measured zero')
        if any(row['method'] == 'tflow' for row in by_name['ambient_sampler_radius_drift']['included']):
            raise AssertionError('t-Flow was incorrectly given a spherical radius-drift comparison')
        for contrast in ('paired_B_minus_A', 'paired_B_minus_C'):
            if {row['metric'] for row in by_name[contrast]['included']} != {'sliced_w1', 'digit_acc', 'fid'}:
                raise AssertionError('A paired metric domain was not drawn')
        from PIL import Image
        files = []
        for figure in manifest['figures']:
            for extension in ('pdf', 'png'):
                path = Path(figure[extension])
                if path.stat().st_size < 1000:
                    raise AssertionError('Empty/corrupt fixture figure: ' + str(path))
                if extension == 'pdf' and not path.read_bytes().startswith(b'%PDF'):
                    raise AssertionError('Invalid PDF signature')
                if extension == 'png':
                    with Image.open(path) as image:
                        dimensions = list(image.size)
                        image.verify()
                    if min(dimensions) < 500:
                        raise AssertionError('Unexpected tiny raster figure')
                files.append({'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size})
        if len(saves) != 14 or not all(row['watermarked'] for row in saves):
            raise AssertionError('Every PDF and PNG must carry the fixture watermark')
        if 'torch' in sys.modules:
            raise AssertionError('Plot fixture validation imported Torch unexpectedly')
        if record['plot_source_sha256'] != sha256(ROOT / 'tools/plot_shared_study.py'):
            raise AssertionError('Plot source changed during validation')
        record.update(status='passed',figure_count=len(names),artifact_count=len(files),figures=sorted(names),
                      fixture_complete_groups=report['complete_method_groups'],omitted_groups=len(manifest['omissions']),
                      plot_manifest=str(output/'fixture_figures/plot_manifest.json'),
                      fixture_report={'path':str(fixture_report),'sha256':sha256(fixture_report)},
                      reference_provenance=report['reference_provenance'],artifacts=files,save_calls=saves,
                      elapsed_s=time.perf_counter()-started)
        write(output / 'check.json', record)
        return record
    except Exception as exc:
        record.update(status='failed',elapsed_s=time.perf_counter()-started,
                      failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        write(output / 'check.json', record)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or CHECK_ROOT / ('job_' + os.environ.get('SLURM_JOB_ID','not_scheduled'))
    result=check(output)
    print(json.dumps({'status':result['status'],'fixture_only':True,'figures':result.get('figure_count'),
                      'artifacts':result.get('artifact_count'),'output':str(output),'elapsed_s':result.get('elapsed_s')}))
