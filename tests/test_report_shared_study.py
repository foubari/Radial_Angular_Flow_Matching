"""Scientific reporting guards, entirely standard library and fixture based."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shared_report', ROOT / 'tools/report_shared_study.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
SEEDS = [8925, 77395, 65457]


def fixture(arm='A', seed=8925):
    cfg = {'condition_id': 'fixture', 'kind': 'vector', 'seeds': SEEDS,
           'data': {'shape': [50, 16]}, 'training': {'steps': 10000},
           'evaluation': {'n_samples': 20, 'sample_seed': 0, 'model_evaluations': 512}}
    cache = {'splits': {key: {'values': {'sha256_contiguous_bytes': key + '-values'},
                             'indices': {'sha256_contiguous_bytes': key + '-indices'}} for key in ('train', 'val', 'test')}}
    dataset = {'splits': {key: {'sha256': key + '-values'} for key in ('train', 'val', 'test')},
               'split_indices': {key: {'sha256': key + '-indices'} for key in ('train', 'val', 'test')}}
    result = {'status': 'complete', 'condition_id': 'fixture', 'method': 'rafm_ang_input', 'arm': arm, 'seed': seed,
        'config': cfg, 'config_sha256': report.config_hash(cfg), 'implementation_sha256': 'implementation',
        'checkpoint': {'step': 10000}, 'metrics': {key: 1. for key in report.required_metrics('vector', arm)},
        'sampler': {'nfe': 512, 'n_batches': 1, 'model_calls_total': 512, 'state_renormalization': False},
        'dataset': dataset, 'parameters': {'total_parameters': 100, 'conditioning_overhead_parameters': 0},
        'radius_drift': {'mean_relative': .001, 'max_relative': .01},
        'sample_artifact': {'sha256': 'sample-sha', 'path': '/not-loaded/samples.pt'},
        'hardware': {'gpu': 'test GPU', 'torch': 'test Torch', 'python': 'test Python', 'cuda': None, 'hip': 'test HIP'}}
    result['metrics']['nfe'] = 512
    return cfg, cache, result


def store_result(directory, result, cfg, *, audit=True):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'result.json'; report.write_json(path, result)
    if audit:
        report.write_json(directory / 'sample_audit.json', {
            'status': 'passed', 'result_sha256': report.sha256(path), 'sample_sha256': 'sample-sha',
            'n_samples': cfg['evaluation']['n_samples'], 'dimension': 16,
            'nfe': 512, 'n_batches': 1, 'model_calls_total': 512,
            'nonfinite_rows': 0, 'nan_rows': 0, 'inf_rows': 0,
            'configured_sample_seed': 0, 'class_counts': None, 'errors': []})


def rows(arm='A', values=(1., 2., 3.)):
    out = []
    with tempfile.TemporaryDirectory() as directory:
        for seed, value in zip(SEEDS, values):
            cfg, cache, result = fixture(arm, seed)
            result['metrics']['sliced_w1'] = value
            target = Path(directory) / str(seed)
            store_result(target, result, cfg)
            out.append(report.collect_seed(cfg, arm, seed, target, cache))
    return out


class SharedReportTests(unittest.TestCase):
    def test_only_exact_three_seed_summary(self):
        for values in ([1], [1, 2], [1, 2, 3, 4], [1, 2, math.nan]):
            with self.assertRaises(ValueError):
                report.summary(values, SEEDS)
        stats = report.summary([1, 2, 3], SEEDS)
        self.assertEqual(stats['mean'], 2)
        self.assertEqual(stats['std_population'], statistics.pstdev([1, 2, 3]))
        self.assertEqual(stats['std_sample'], 1)

    def test_complete_valid_triplet_aggregates(self):
        records = rows()
        self.assertEqual([row['status'] for row in records], ['complete'] * 3)
        group = report.aggregate_group(records, SEEDS)
        self.assertEqual(group['status'], 'complete')
        self.assertEqual(group['aggregate']['sliced_w1']['mean'], 2)
        self.assertEqual(group['radius_drift_summary']['mean_relative']['mean'], .001)

    def test_no_survivor_mean_after_failure_or_missing_seed(self):
        for status in ('failed', 'missing', 'blocked', 'incompatible', 'awaiting_sample_audit'):
            records = rows(); records[1]['status'] = status
            group = report.aggregate_group(records, SEEDS)
            self.assertEqual(group['status'], 'incomplete')
            self.assertEqual(group['aggregate'], {})
            self.assertEqual(group['parameter_summary'], {})

    def test_no_mean_after_mixed_source_or_dataset(self):
        for key in ('source', 'implementation_sha256', 'dataset_manifest', 'config_sha256'):
            records = rows(); records[1][key] = 'different'
            group = report.aggregate_group(records, SEEDS)
            self.assertEqual(group['status'], 'incompatible')
            self.assertEqual(group['aggregate'], {})

    def test_unequal_metric_sets_not_silently_intersected(self):
        records = rows(); records[1]['metrics']['new_metric'] = .2
        group = report.aggregate_group(records, SEEDS)
        self.assertEqual(group['status'], 'incompatible')
        self.assertEqual(group['aggregate'], {})

    def test_nonfinite_or_missing_metric_invalidates_complete_record(self):
        for value in (None, float('inf'), float('nan')):
            cfg, cache, result = fixture(); result['metrics']['sliced_w1'] = value
            self.assertTrue(report.validate_result(result, cfg, 'A', SEEDS[0], cache))
        cfg, cache, result = fixture(); del result['metrics']['angular_sw_bin3']
        self.assertIn('missing/nonfinite required metric: angular_sw_bin3', report.validate_result(result, cfg, 'A', SEEDS[0], cache))

    def test_checkpoint_and_actual_nfe_are_checked(self):
        cfg, cache, result = fixture(); result['checkpoint']['step'] = 5000
        self.assertTrue(report.validate_result(result, cfg, 'A', SEEDS[0], cache))
        cfg, cache, result = fixture(); result['sampler']['model_calls_total'] = 128
        self.assertIn('actual network call count missing or inconsistent', report.validate_result(result, cfg, 'A', SEEDS[0], cache))

    def test_sample_audit_required_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, cache, result = fixture(); path = Path(directory)
            store_result(path, result, cfg, audit=False)
            self.assertEqual(report.collect_seed(cfg, 'A', 8925, path, cache)['status'], 'awaiting_sample_audit')
            store_result(path, result, cfg)
            audit = report.read_json(path / 'sample_audit.json'); audit['sample_sha256'] = 'wrong'
            report.write_json(path / 'sample_audit.json', audit)
            self.assertEqual(report.collect_seed(cfg, 'A', 8925, path, cache)['status'], 'incompatible')

    def test_failed_results_remain_failed_without_average(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, cache, result = fixture(); result['status'] = 'failed'; result['failure'] = {'message': 'nonfinite loss'}
            store_result(directory, result, cfg, audit=False)
            row = report.collect_seed(cfg, 'A', 8925, Path(directory), cache)
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(row['failure']['message'], 'nonfinite loss')

    def test_paired_deltas_use_same_seeds_and_preserve_negative_outcome(self):
        groups = {arm: report.aggregate_group(rows(arm, values), SEEDS) for arm, values in [('A', (1., 2., 3.)), ('B', (2., 3., 4.)), ('C', (3., 2., 1.))]}
        result = report.paired(groups, 'B', 'A', SEEDS)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['metrics']['sliced_w1']['values'], [1., 1., 1.])
        self.assertEqual(result['metrics']['sliced_w1']['seeds_favor_left'], 0)
        self.assertTrue(result['paired_runtime_environment_verified'])
        groups['A']['seeds'][0]['dataset_manifest'] = {'different': True}
        self.assertEqual(report.paired(groups, 'B', 'A', SEEDS)['status'], 'incompatible')

    def test_partial_pair_never_aggregates(self):
        groups = {'A': report.aggregate_group(rows(), SEEDS), 'B': {'status': 'incomplete'}}
        self.assertEqual(report.paired(groups, 'B', 'A', SEEDS)['metrics'], {})

    def test_reference_missing_identity_never_eligible(self):
        reference = report.fixed_gain_reference(None, None, Path('/missing'), Path('/missing'), None)
        self.assertFalse(reference['eligible_for_prepared_protocol'])
        self.assertEqual(reference['status'], 'unresolved')
        self.assertEqual(reference['historical_fixed_accuracy']['mean'], .810)

    def test_implementation_and_old_attempt_failures_stay_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            abc = root / 'abc/v1'
            tflow_v1 = root / 'tflow/v1'
            tflow_v2 = root / 'tflow/v2'
            report.write_json(abc / 'validation_failures/765729/failure.json', {
                'phase': 'unit_checks', 'status': 'failed', 'passed': 46, 'failed': 4,
                'training_or_benchmark_results_affected': False})
            report.write_json(tflow_v1 / 'execution_failure_summary.json', {
                'status': 'cancelled_after_startup_failure', 'optimizer_updates_observed': 0,
                'checkpoint_files': [], 'failures': [{'condition': 'fixture'}]})
            report.write_json(tflow_v1 / 'tuning/fixture/candidate_00/failed.json', {'status': 'failed'})
            report.write_json(tflow_v2 / 'executions/current_worker.json', {'status': 'running'})
            operation = report.collect_operations(abc, tflow_v2)
            self.assertEqual(len(operation['implementation_check_failures']), 1)
            self.assertFalse(operation['implementation_check_failures'][0]['record']['training_or_benchmark_results_affected'])
            self.assertEqual(len(operation['previous_tflow_attempts']), 1)
            previous = operation['previous_tflow_attempts'][0]
            self.assertFalse(previous['counts_as_current_final_seed_failure'])
            self.assertEqual(previous['record']['optimizer_updates_observed'], 0)
            self.assertEqual(len(previous['candidate_failure_records']), 1)
            self.assertEqual(len(operation['current_execution_records']['tflow']), 1)
            self.assertEqual(operation['scheduler_snapshots'], [])

    def test_audio_reference_has_explicit_separate_row_and_full_metrics(self):
        values = {'digit_acc': .8066666666666666, 'energy_KS': .0218333333333,
                  'cov>q95': .05, 'cov>q99': .0145}
        data = {'conditions': [{'condition_id': 'audiomnist_stft', 'methods': {
            arm: {'status': 'incomplete', 'aggregate': {}} for arm in ('A', 'B', 'C', 'tflow')}}],
            'fixed_spherical_gain_reference': {'eligible_for_prepared_protocol': True, 'backend_compatibility': {'verified': True},
                'training_seeds': [8925, 1234, 7],
                'measured_full_precision': {key: {'mean': value, 'std': .007352248333 if key == 'digit_acc' else 0.,
                                                  'vals': [value] * 3} for key, value in values.items()}}}
        rows = report.audio_comparison_rows(data)
        self.assertEqual(len(rows), 5)
        reference = rows[-1]
        self.assertEqual(reference['status'], 'verified_checkpoint_reference')
        self.assertIn('not new A or historical paper value', reference['role'])
        self.assertEqual(set(reference['metrics']), set(values))
        self.assertEqual(reference['metrics']['digit_acc']['mean'], values['digit_acc'])
        self.assertEqual(rows[0]['method'], 'A')
        self.assertEqual(rows[0]['metrics'], {})
        data['fixed_spherical_gain_reference']['eligible_for_prepared_protocol'] = False
        self.assertEqual(len(report.audio_comparison_rows(data)), 4)

    def test_audio_backend_must_measure_zero_changes_and_matching_energy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / 'aggregate.json'; report.write_json(aggregate, {'status': 'baseline_mismatch'})
            cfg = {'seeds': [8925, 1234, 7], 'evaluation': {'n_samples': 2000}}
            reference = {'path': str(aggregate)}
            path = root / 'audit.json'
            report.attach_reference_backend(reference, cfg, path)
            self.assertFalse(reference['backend_compatibility']['verified'])
            audit = {'status': 'passed', 'config_sha256': report.config_hash(cfg),
                'aggregate': {'sha256': report.sha256(aggregate)},
                'totals': {'old_vs_new_prediction_disagreements': 0, 'postrescale_prediction_disagreements': 0},
                'runs': [{'seed': seed, 'n': 2000, 'invariance': {'passed': True, 'prediction_disagreements': 0},
                          'baseline': {'comparison': {'prediction_disagreements_vs_cached': 0, 'energy_ks_tail_metrics_exactly_equal': True}},
                          'posthoc': {'comparison': {'prediction_disagreements_vs_cached': 0, 'energy_ks_tail_metrics_exactly_equal': True}}}
                         for seed in cfg['seeds']]}
            report.write_json(path, audit)
            report.attach_reference_backend(reference, cfg, path)
            self.assertTrue(reference['backend_compatibility']['verified'])
            audit['runs'][1]['posthoc']['comparison']['prediction_disagreements_vs_cached'] = 1
            report.write_json(path, audit)
            report.attach_reference_backend(reference, cfg, path)
            self.assertFalse(reference['backend_compatibility']['verified'])
            audit['runs'][1]['posthoc']['comparison']['prediction_disagreements_vs_cached'] = 0
            audit['runs'][0]['baseline']['comparison']['energy_ks_tail_metrics_exactly_equal'] = False
            report.write_json(path, audit)
            report.attach_reference_backend(reference, cfg, path)
            self.assertFalse(reference['backend_compatibility']['verified'])

    def test_plotting_skips_without_inventing_results(self):
        specification = importlib.util.spec_from_file_location('shared_plot', ROOT / 'tools/plot_shared_study.py')
        plot = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(plot)
        with tempfile.TemporaryDirectory() as directory:
            result = plot.plot_report({'conditions': [], 'complete_method_groups': 0}, directory)
            self.assertEqual(result['status'], 'no_complete_triplets')
            self.assertEqual(result['figures'], [])
            self.assertFalse(list(Path(directory).glob('*.png')))
            self.assertFalse(list(Path(directory).glob('*.pdf')))
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'Slurm compute allocation'):
                plot.plot_report({'conditions': [], 'complete_method_groups': 1}, directory)

    def test_import_does_not_load_numeric_packages(self):
        subprocess.run([sys.executable, '-c', 'import runpy,sys; runpy.run_path("tools/report_shared_study.py",run_name="audit"); assert "torch" not in sys.modules; assert "numpy" not in sys.modules'], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
