"""Scientific findings and exact continuation-cost accounting, stdlib fixtures."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('findings', ROOT / 'tools/summarize_shared_study.py')
findings = importlib.util.module_from_spec(spec); spec.loader.exec_module(findings)
SEEDS = [8925, 77395, 65457]


def condition():
    case = {'condition_id': 'fixture', 'kind': 'vector', 'seeds': SEEDS, 'config_sha256': 'cfg',
            'blocking_issues': [], 'comparison_label': 'new_matched_realization', 'training': {'steps': 10000, 'precision': 'float32'},
            'B_C_parameter_equality': 'passed', 'methods': {}}
    for method, values in [('A', (3., 4., 5.)), ('B', (2., 3., 4.)), ('C', (1., 2., 3.)), ('tflow', (4., 5., 6.))]:
        rows = []
        for seed, value in zip(SEEDS, values):
            rows.append({'seed': seed, 'status': 'complete', 'config_sha256': 'cfg', 'dataset_manifest': {'train': 'hash'},
                'metrics': {'sliced_w1': value, **{key: value for key in findings.ANGULAR},
                            'total_train_time_s': 20 if method == 'B' else 10, 'sample_time_s': 4 if method == 'B' else 2},
                'parameters': {'total_parameters': 110 if method in ('B', 'C') else 100, 'original_backbone_parameters': 100,
                               'conditioning_overhead_parameters': 10 if method in ('B', 'C') else 0,
                               'conditioning_overhead_fraction': .1 if method in ('B', 'C') else 0},
                'hardware': {'gpu': 'MI210', 'torch': 'test', 'python': 'test', 'hip': 'test', 'cuda': None}})
        case['methods'][method] = {'status': 'complete', 'seeds': rows}
    return case


def report(case):
    return {'study_id': 'fixture', 'status': 'complete', 'conditions': [case], 'expected_final_runs': 12,
            'seed_status_counts': {'complete': 12}, 'complete_conditions': [case['condition_id']],
            'fixed_spherical_gain_reference': {}, 'operations': {}}


def selection():
    def trial(index, steps, elapsed):
        return {'candidate_index': index, 'steps': steps, 'status': 'complete', 'source': {'nu': 3 + index},
                'training_time_s': elapsed, 'sample_time_s': 1., 'config_sha256': 'cfg', 'implementation_sha256': 'impl'}
    return {'status': 'frozen', 'selection_split': 'validation', 'test_data_used_for_selection': False,
            'config_sha256': 'cfg', 'implementation_sha256': 'impl',
            'stage1_trials': [trial(i, 500, 10.) for i in range(9)],
            'finalist_trials': [trial(2, 1000, 15.), trial(5, 1000, 17.)], 'selected': {'nu': 5}}


class FindingsTests(unittest.TestCase):
    def test_answers_are_measured_not_canned(self):
        result = findings.build_findings(report(condition()))
        self.assertEqual(result['tallies']['B-A']['mean_outcomes'], {'improved': 1, 'worsened': 0, 'tied': 0})
        self.assertEqual(result['tallies']['B-C']['mean_outcomes'], {'improved': 0, 'worsened': 1, 'tied': 0})
        self.assertIn('improves in 1', result['answers']['1'])
        self.assertIn('worsens in 1', result['answers']['2'])
        self.assertEqual(result['tallies']['tflow-A']['mean_outcomes']['worsened'], 1)

    def test_mean_effect_does_not_hide_mixed_seeds(self):
        case = condition()
        for row, value in zip(case['methods']['B']['seeds'], [2., 5., 4.]):
            row['metrics']['sliced_w1'] = value
        effect = findings.comparison(case, 'B', 'A')['primary']
        self.assertEqual(effect['mean_outcome'], 'improved')
        self.assertEqual(effect['consistency'], 'mixed_or_tied_seeds')
        self.assertEqual(effect['seed_outcome_counts'], {'improved': 2, 'worsened': 1})

    def test_partial_or_failed_triplet_has_no_effect(self):
        for change in ('failed', 'missing', 'incompatible'):
            case = condition(); case['methods']['B']['seeds'][0]['status'] = change
            pair = findings.comparison(case, 'B', 'A')
            self.assertEqual(pair['status'], 'unavailable')
            self.assertEqual(pair['primary'], {})

    def test_identity_and_parameter_mismatch_block_comparison(self):
        case = condition(); case['methods']['B']['seeds'][1]['dataset_manifest'] = {'different': True}
        self.assertEqual(findings.comparison(case, 'B', 'A')['status'], 'unavailable')
        case = condition(); case['B_C_parameter_equality'] = 'failed'
        self.assertEqual(findings.comparison(case, 'B', 'C')['status'], 'unavailable')
        case = condition(); case['blocking_issues'] = ['missing_identity']
        self.assertEqual(findings.rankings(case)['methods'], {})

    def test_primary_metrics_have_correct_direction_without_pooling(self):
        case = condition(); case['kind'] = 'audio'
        for method, group in case['methods'].items():
            for row in group['seeds']:
                row['metrics']['digit_acc'] = .8 if method == 'B' else .7
        effect = findings.comparison(case, 'B', 'A')['primary']
        self.assertEqual(effect['better_direction'], 'higher')
        self.assertEqual(effect['mean_outcome'], 'improved')
        self.assertAlmostEqual(effect['delta_left_minus_right']['mean'], .1)
        self.assertEqual(findings.PRIMARY['image'], ('fid', 'lower'))

    def test_raw_runtime_ratios_require_matching_hardware_and_precision(self):
        case = condition(); pair = findings.comparison(case, 'B', 'A')
        ratio = pair['runtime']['ratios']['total_train_time_s']
        self.assertEqual(ratio['per_seed_ratio_left_over_right']['mean'], 2.)
        case['methods']['B']['seeds'][0]['hardware']['gpu'] = 'A100'
        self.assertEqual(findings.comparison(case, 'B', 'A')['runtime']['status'], 'unresolved_hardware_or_precision')
        case = condition(); del case['training']['precision']
        self.assertEqual(findings.comparison(case, 'B', 'A')['runtime']['ratios'], {})

    def test_zero_time_is_not_divided_or_estimated(self):
        case = condition(); case['methods']['A']['seeds'][0]['metrics']['sample_time_s'] = 0.
        runtime = findings.comparison(case, 'B', 'A')['runtime']
        self.assertEqual(runtime['ratios']['sample_time_s']['status'], 'unavailable')

    def test_tuning_cost_subtracts_finalists_initial_cumulative_times(self):
        case = condition(); case['tflow_tuning'] = {'records': [{'record': selection()}]}
        cost = findings.tuning_cost(case)
        self.assertEqual(cost['total_training_loop_s'], 102.)  # 9*10 + (15-10) + (17-10)
        self.assertEqual(cost['validation_sampling_s'], 11.)
        self.assertEqual(cost['training_updates'], 5500)
        self.assertEqual(cost['full_training_equivalents'], .55)

    def test_unknown_inconsistent_or_partial_tuning_is_not_estimated(self):
        for change in ('missing_time', 'shorter_time', 'wrong_source', 'test_used', 'missing_trial'):
            frozen = selection()
            if change == 'missing_time': del frozen['stage1_trials'][0]['training_time_s']
            if change == 'shorter_time': frozen['finalist_trials'][0]['training_time_s'] = 5.
            if change == 'wrong_source': frozen['finalist_trials'][0]['source'] = {'nu': 99}
            if change == 'test_used': frozen['test_data_used_for_selection'] = True
            if change == 'missing_trial': frozen['stage1_trials'].pop()
            case = condition(); case['tflow_tuning'] = {'records': [{'record': frozen}]}
            self.assertEqual(findings.tuning_cost(case)['status'], 'unavailable')

    def test_rankings_are_limited_to_available_complete_methods(self):
        case = condition(); case['methods']['C']['status'] = 'incomplete'
        rank = findings.rankings(case)
        self.assertEqual(rank['order_by_measured_mean'], ['B', 'A', 'tflow'])
        self.assertEqual(rank['missing_methods'], ['C'])

    def test_audio_reference_requires_completed_executed_compatibility(self):
        case = condition(); case['kind'] = 'audio'; case['condition_id'] = 'audiomnist_stft'
        for method, group in case['methods'].items():
            for row in group['seeds']:
                row['metrics'].update(digit_acc=.8, energy_KS=.02, **{'cov>q95': .05, 'cov>q99': .01})
        data = report(case)
        ref = {'eligible_for_prepared_protocol': True, 'training_seeds': SEEDS,
               'measured_full_precision': {key: {'vals': [value] * 3} for key, value in {'digit_acc': .8067, 'energy_KS': .0218, 'cov>q95': .05, 'cov>q99': .0145}.items()},
               'reproduction_limitation': 'Measured .8067 differs from historical .810', 'new_audio_execution_compatibility': []}
        data['fixed_spherical_gain_reference'] = ref
        self.assertEqual(findings.audio_reference_comparison(data)['status'], 'unavailable')
        ref['new_audio_execution_compatibility'] = [{'method': 'A', 'seed': seed, 'status': 'verified'} for seed in SEEDS]
        self.assertEqual(findings.audio_reference_comparison(data)['status'], 'unavailable')
        ref['backend_compatibility'] = {'verified': True}
        self.assertEqual(findings.audio_reference_comparison(data)['status'], 'unavailable')
        ref['current_backend_reference'] = {'usable_for_comparison': True,
            'aggregate_metrics': {'posthoc': {key: {'values': value['vals']} for key, value in ref['measured_full_precision'].items()}}}
        # Archived values must never replace the newly measured backend values.
        ref['measured_full_precision']['digit_acc']['vals'] = [.999] * 3
        result = findings.audio_reference_comparison(data)
        self.assertEqual(result['best_digit_accuracy_methods'], ['fixed_spherical_empirical_gain_reference'])
        self.assertEqual(set(result['methods']), {'A', 'fixed_spherical_empirical_gain_reference'})
        self.assertTrue(result['not_a_training_runtime_comparison'])

    def test_failed_seed_and_blockers_are_preserved_in_findings(self):
        case = condition(); case['blocking_issues'] = ['missing_tensor']; case['methods']['B']['seeds'][0]['status'] = 'failed'
        data = report(case); data['status'] = 'incomplete'; data['complete_conditions'] = []
        result = findings.build_findings(data)
        self.assertEqual(result['status'], 'partial_suite')
        self.assertEqual(result['coverage']['blocked_conditions'][0]['issues'], ['missing_tensor'])
        self.assertEqual(len(result['current_final_failures']), 1)

    def test_output_is_self_contained_and_import_is_stdlib(self):
        result = findings.build_findings(report(condition()))
        with tempfile.TemporaryDirectory() as directory:
            findings.write_findings(result, directory)
            self.assertTrue((Path(directory) / 'findings.json').is_file())
            text = (Path(directory) / 'findings.md').read_text()
            self.assertIn('improves in 1', text)
            self.assertIn('B-C', text)
            self.assertIn('no new theoretical guarantee', text)
        subprocess.run([sys.executable, '-c', 'import runpy,sys;runpy.run_path("tools/summarize_shared_study.py",run_name="inspection");assert "torch" not in sys.modules;assert "numpy" not in sys.modules'], cwd=ROOT, check=True)

    def test_collector_then_findings_manifest_pins_final_recursive_bytes(self):
        spec = importlib.util.spec_from_file_location('collector_fixture', ROOT / 'tools/report_shared_study.py')
        collector = importlib.util.module_from_spec(spec); spec.loader.exec_module(collector)
        data = {'study_id': 'fixture', 'status': 'incomplete', 'conditions': [], 'expected_final_runs': 0,
                'seed_status_counts': {}, 'complete_conditions': [], 'complete_method_groups': 0,
                'fixed_spherical_gain_reference': {'status': 'missing', 'issues': [],
                    'reproduction_limitation': 'Historical accuracy remains distinct.'}, 'operations': {}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'findings.json').write_text('{"stale": true}\n')
            collector.write_outputs(data, directory)
            initial = json.loads((directory / 'report_files.json').read_text())
            stale_hash = next(item['sha256'] for item in initial['files'] if Path(item['path']).name == 'findings.json')
            figures = directory / 'figures'; figures.mkdir()
            # Binary fixtures stand in for exported figures; no plotting/ML import.
            (figures / 'comparison.pdf').write_bytes(b'%PDF-1.4 fixture\n')
            (figures / 'comparison.png').write_bytes(b'\x89PNG\r\n\x1a\nfixture')
            (figures / 'plot_manifest.json').write_text('{"status": "fixture"}\n')
            (figures / '.partial.png').write_bytes(b'temporary')
            (directory / 'scratch.tmp').write_text('temporary')
            subprocess.run([sys.executable, 'tools/summarize_shared_study.py', '--report', str(directory / 'report.json')],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            manifest = json.loads((directory / 'report_files.json').read_text())
            files = {item['relative_path']: item for item in manifest['files']}
            for name in ('report.json', 'findings.json', 'findings.md', 'figures/comparison.pdf',
                         'figures/comparison.png', 'figures/plot_manifest.json'):
                raw = (directory / name).read_bytes()
                self.assertEqual(files[name]['sha256'], hashlib.sha256(raw).hexdigest())
                self.assertEqual(files[name]['size_bytes'], len(raw))
            self.assertNotEqual(files['findings.json']['sha256'], stale_hash)
            self.assertEqual(manifest['input_report']['sha256'], files['report.json']['sha256'])
            self.assertNotIn('report_files.json', files)
            self.assertNotIn('scratch.tmp', files)
            self.assertNotIn('figures/.partial.png', files)

    def test_changed_input_report_aborts_without_replacing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            report_path = directory / 'report.json'; report_path.write_text('{"old": true}\n')
            result = {'input_report': {'path': str(report_path), **findings.file_fingerprint(report_path)}}
            target = directory / 'report_files.json'; target.write_text('original manifest\n')
            report_path.write_text('{"changed": true}\n')
            with self.assertRaisesRegex(ValueError, 'report.json changed'):
                findings.finalize_report_manifest(result, directory)
            self.assertEqual(target.read_text(), 'original manifest\n')


if __name__ == '__main__':
    unittest.main()
