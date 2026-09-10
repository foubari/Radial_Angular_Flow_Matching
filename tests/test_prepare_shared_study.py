"""No tensor imports: authorization, protocol selection and immutable-cache guards."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_shared_study', ROOT / 'tools/prepare_shared_study.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class SharedStudyPreparationTests(unittest.TestCase):
    def setUp(self):
        self.plan = prep.make_plan()
        self.cases = {item['condition_id']: item for item in self.plan['drafts']}

    def test_complete_scope_and_explicit_blockers(self):
        self.assertEqual(len(self.cases), 28)
        self.assertEqual(len(self.plan['materializable_conditions']), 26)
        self.assertEqual(len(self.plan['new_synthetic_realizations']), 20)
        self.assertEqual(set(self.plan['blocked_conditions']), {'piv_d32', 'imagenette_dcae'})
        for condition in self.plan['blocked_conditions']:
            self.assertIsNone(self.cases[condition]['data']['split'].get('file', {}).get('sha256'))

    def test_paper_batch_sizes_preserved(self):
        for condition, cfg in self.cases.items():
            if condition.startswith(('student_t_', 'piv_')):
                self.assertEqual(cfg['training']['batch_size'], 256)
            elif cfg['kind'] == 'vector':
                self.assertEqual(cfg['training']['batch_size'], 2048 if condition == 'weather_au_wind' else 4096)
        self.assertEqual(self.cases['audiomnist_stft']['training']['batch_size'], 32)
        self.assertEqual(self.cases['imagenette_dcae']['training']['batch_size'], 64)

    def test_training_budgets_seeds_and_no_BC_tuning(self):
        for cfg in self.cases.values():
            downstream = cfg['kind'] != 'vector'
            self.assertEqual(cfg['seeds'], [8925, 1234, 7] if downstream else [8925, 77395, 65457])
            self.assertEqual(cfg['training']['steps'], {'vector': 10000, 'audio': 24000, 'image': 40000}[cfg['kind']])
            self.assertEqual(cfg['study']['B_C_additional_tuning_steps'], 0)
            self.assertTrue(cfg['study']['new_A_rerun'])
            self.assertFalse(cfg['provenance']['historical_result_reuse'])

    def test_no_runnable_config_before_hashes(self):
        for cfg in self.cases.values():
            self.assertEqual(cfg['protocol_status'], 'blocked')
            self.assertTrue(cfg['blocking_issues'])
            self.assertIsNone(cfg['shared_cache']['manifest_sha256'])

    def test_original_configs_not_mutated(self):
        original = json.loads((ROOT / 'configs/tflow/prepared/student_t_d16_df3.0_cor.json').read_text())
        before = copy.deepcopy(original)
        prep.make_draft(original, prep.DEFAULT_DATA)
        self.assertEqual(original, before)
        self.assertIsNone(original['data']['input']['path'])

    def test_synthetic_generation_seed_is_condition_stable(self):
        specs = [cfg['provenance']['generation'] for cfg in self.cases.values() if cfg['provenance']['generation']]
        seeds = [value['sample_seed'] for value in specs if value['family'] != 'aniso_sweep']
        self.assertEqual(len(seeds), len(set(seeds)))
        for condition in self.plan['new_synthetic_realizations']:
            cfg = self.cases[condition]
            self.assertEqual(cfg['provenance']['generation'], prep.generation_spec(condition, cfg['data']['shape']))
            self.assertEqual(cfg['evaluation']['sample_seed'], 0)
            self.assertEqual(cfg['provenance']['comparison_label'], 'new_matched_realization')

    def test_target_student_t_distinct_from_multivariate_source(self):
        cfg = self.cases['student_t_d16_df1.5_cor']
        self.assertIn('independent univariate', cfg['provenance']['generation']['target_noise'])
        self.assertEqual(cfg['provenance']['generation']['matrix_seed'], 42)
        self.assertEqual(cfg['rafm_sampling_source']['kind'], 'radial_empirical_ecdf')
        self.assertEqual(cfg['tuning']['nu'], [3, 5, 7])

    def test_existing_inputs_and_splits_not_substituted(self):
        for condition in ['finance_ff49', 'weather_au_wind', 'piv_d16', 'piv_d64', 'piv_d256', 'audiomnist_stft']:
            original = json.loads((ROOT / 'configs/tflow/prepared' / (condition + '.json')).read_text())
            cfg = self.cases[condition]
            self.assertEqual(cfg['data']['input'], original['data']['input'])
            self.assertEqual(cfg['provenance']['original_split_spec'], original['data']['split'])
            self.assertEqual(cfg['evaluation']['sample_seed'], original['evaluation']['sample_seed'])
        audio = self.cases['audiomnist_stft']['data']
        self.assertNotEqual(audio['input']['path'], audio['external_test']['path'])
        self.assertEqual([audio['split'][x] for x in ['n_train', 'n_val', 'n_test']], [10200, 1800, 0])

    def test_login_materialization_aborts_before_numeric_imports(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'Slurm compute allocation'):
                prep.materialize(self.cases['finance_ff49'])

    def test_existing_manifest_checksum_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            asset = path / 'train.pt'
            asset.write_bytes(b'original')
            manifest = {'status': 'complete', 'preparation_identity_sha256': 'identity',
                        'assets': [{'path': str(asset), 'sha256': prep.sha256(asset)}]}
            (path / 'manifest.json').write_text(json.dumps(manifest))
            prep.validate_existing_manifest(path, 'identity')
            asset.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                prep.validate_existing_manifest(path, 'identity')
            asset.unlink()
            with self.assertRaisesRegex(ValueError, 'missing or checksum mismatch'):
                prep.validate_existing_manifest(path, 'identity')

    def test_existing_manifest_different_identity_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'manifest.json').write_text(json.dumps({'status': 'complete', 'preparation_identity_sha256': 'old', 'assets': []}))
            with self.assertRaisesRegex(ValueError, 'different preparation identity'):
                prep.validate_existing_manifest(path, 'new')

    def test_import_is_stdlib_only(self):
        program = 'import runpy,sys; runpy.run_path("tools/prepare_shared_study.py",run_name="inspection"); assert "torch" not in sys.modules; assert "numpy" not in sys.modules'
        subprocess.run([sys.executable, '-c', program], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
