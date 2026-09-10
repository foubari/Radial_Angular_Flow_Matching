"""Export measured shared-study figures; imports matplotlib only inside Slurm.

No plot is rendered until at least one three-seed method group is complete.
All omitted, failed, and blocked conditions remain in the figure manifest.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import textwrap


def plot_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    omissions = []
    for condition in report['conditions']:
        for method, group in condition['methods'].items():
            if group['status'] != 'complete':
                omissions.append({'condition': condition['condition_id'], 'method': method,
                                  'status': group['status'], 'seed_status_counts': group['seed_status_counts'],
                                  'blocking_issues': condition['blocking_issues']})
    manifest = {'status': 'no_complete_triplets', 'figures': [], 'omissions': omissions,
                'policy': 'Plot only measured complete three-seed groups. Error bars are population SD; small dots are individual seeds. No imputed values.',
                'timing_note': 'Current measured times, not historical timings; inspect per-seed hardware/software before interpreting speed differences.'}
    if not report['complete_method_groups']:
        (output / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Scientific figure rendering with numerical libraries must run in a Slurm compute allocation')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {'A': '#2878b5', 'B': '#24975c', 'C': '#e1812c', 'tflow': '#8756a3'}
    blocked = sorted({row['condition'] for row in omissions if row['blocking_issues']})
    failed = sorted({row['condition'] + '/' + row['method'] for row in omissions
                     if any(row['seed_status_counts'].get(s, 0) for s in ('failed', 'incompatible', 'unreadable', 'unrecognized_status'))})
    footer = ('Only complete three-seed groups shown. Blocked: ' + (', '.join(blocked) or 'none') +
              '. Failed/incompatible groups: ' + (', '.join(failed) or 'none recorded') +
              '. All pending groups are listed in plot_manifest.json; published results are unchanged.')
    footer_lines = textwrap.wrap(footer, 155)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})

    def finish(fig, name, title, included, caption):
        fig.suptitle(title, fontsize=12, y=.995)
        bottom = min(.42, .07 + .025 * len(footer_lines))
        fig.tight_layout(rect=(0, bottom, 1, .965))
        fig.text(.01, .015, '\n'.join(footer_lines), fontsize=7, va='bottom')
        for suffix in ('pdf', 'png'):
            fig.savefig(output / (name + '.' + suffix), dpi=180)
        plt.close(fig)
        manifest['figures'].append({'name': name, 'included': included, 'caption': caption + ' ' + footer,
                                   'pdf': str(output / (name + '.pdf')), 'png': str(output / (name + '.png'))})

    def labels(axis, ids):
        axis.set_xticks(range(len(ids)))
        axis.set_xticklabels(ids, rotation=75, ha='right', fontsize=7)
        axis.grid(axis='y', alpha=.2)
        axis.margins(x=.035)

    # Paired differences answer the input and radius-conditioning questions.
    for contrast in ('B-A', 'B-C'):
        panels = []
        for kind, metric in [('vector', 'sliced_w1'), ('audio', 'digit_acc'), ('image', 'fid')]:
            rows = [(c, c['paired_contrasts'][contrast]['metrics'][metric]) for c in report['conditions']
                    if c['kind'] == kind and c['paired_contrasts'][contrast]['status'] == 'complete'
                    and metric in c['paired_contrasts'][contrast]['metrics']]
            if rows:
                panels.append((kind, metric, rows))
        if not panels:
            continue
        fig, axes = plt.subplots(len(panels), 1, figsize=(max(10, max(len(p[2]) for p in panels) * .52), 4.5 * len(panels)), squeeze=False)
        included = []
        for axis, (kind, metric, rows) in zip(axes[:, 0], panels):
            ids = [c['condition_id'] for c, value in rows]
            for index, (condition, value) in enumerate(rows):
                axis.errorbar(index, value['mean'], yerr=value['std_population'], fmt='o', color=colors['B'], capsize=3)
                axis.scatter([index - .09, index, index + .09], value['values'], s=13, color='black', alpha=.55, zorder=3)
                included.append({'condition': condition['condition_id'], 'contrast': contrast, 'metric': metric})
            axis.axhline(0, color='gray', linewidth=1)
            axis.set_ylabel(contrast + ' ' + metric)
            axis.set_title(kind + ': ' + ('positive favors B' if metric == 'digit_acc' else 'negative favors B'))
            labels(axis, ids)
        finish(fig, 'paired_' + contrast.replace('-', '_minus_'), 'Paired input-parameterization differences', included,
               'Mean paired delta with population SD and each training-seed delta. Deltas compare matching cached data, protocol and training seeds; no significance is asserted.')

    # Original and new models, including t-Flow, retain their own measured values.
    panels = []
    for kind, metric in [('vector', 'sliced_w1'), ('audio', 'digit_acc'), ('image', 'fid')]:
        rows = [c for c in report['conditions'] if c['kind'] == kind and any(metric in g['aggregate'] for g in c['methods'].values())]
        if rows:
            panels.append((kind, metric, rows))
    if panels:
        fig, axes = plt.subplots(len(panels), 1, figsize=(max(11, max(len(p[2]) for p in panels) * .58), 4.7 * len(panels)), squeeze=False)
        included = []
        for axis, (kind, metric, rows) in zip(axes[:, 0], panels):
            for index, condition in enumerate(rows):
                for offset, (method, group) in zip((-.24, -.08, .08, .24), condition['methods'].items()):
                    value = group['aggregate'].get(metric)
                    if value is None:
                        continue
                    x = index + offset
                    axis.errorbar(x, value['mean'], yerr=value['std_population'], fmt='o', color=colors[method], capsize=2)
                    axis.scatter([x - .035, x, x + .035], value['values'], s=8, color=colors[method], alpha=.5)
                    included.append({'condition': condition['condition_id'], 'method': method, 'metric': metric})
            if kind == 'audio':
                ref = report['fixed_spherical_gain_reference']
                if ref.get('eligible_for_prepared_protocol'):
                    value = ref['measured_full_precision']['digit_acc']
                    axis.errorbar(.4, value['mean'], yerr=value['std'], fmt='*', color='#555555', markersize=10, capsize=3)
                    included.append({'condition': 'audiomnist_stft', 'method': 'fixed_spherical_empirical_gain_reference', 'metric': 'digit_acc', 'historical_accuracy_mismatch_preserved': True})
            axis.set_ylabel(metric)
            axis.set_title(kind)
            if kind != 'audio':
                axis.set_yscale('symlog', linthresh=1e-4)
            labels(axis, [c['condition_id'] for c in rows])
        handles = [Line2D([], [], marker='o', linestyle='', color=color, label=method) for method, color in colors.items()]
        handles.append(Line2D([], [], marker='*', linestyle='', color='#555555', label='Fixed-spherical + gain reference'))
        axes[0, 0].legend(handles=handles, ncol=5, fontsize=8)
        finish(fig, 'quality_tflow_and_rafm_inputs', 'Measured quality at matched final training budgets', included,
               'Complete three-seed method means and population SD. Vector/image axes use symmetric-log scaling with linear threshold 1e-4. The audio reference retains its measured approximately 0.8067 accuracy versus reported 0.810 discrepancy; its historical training runtime is not compared.')

    # Downstream content and calibration share one explicitly labelled reference.
    audio_rows = report.get('audio_comparison_rows', [])
    if any(row['status'] == 'complete' for row in audio_rows):
        rows = [row for row in audio_rows if row['status'] in ('complete', 'verified_checkpoint_reference')]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
        included = []
        metrics = [('digit_acc', 'Digit accuracy; higher is better', None),
                   ('energy_KS', 'Energy KS; lower is better', 0.),
                   ('cov>q95', 'Coverage above external-test q95', .05),
                   ('cov>q99', 'Coverage above external-test q99', .01)]
        for axis, (metric, label, target) in zip(axes.flat, metrics):
            names = []
            for index, row in enumerate(rows):
                value = row['metrics'].get(metric)
                names.append('Fixed-spherical + gain\ncheckpoint reference' if row['status'] == 'verified_checkpoint_reference' else row['method'])
                if value is None:
                    continue
                reference = row['status'] == 'verified_checkpoint_reference'
                color = '#555555' if reference else colors[row['method']]
                axis.errorbar(index, value['mean'], yerr=value['std_population'], fmt='*' if reference else 'o', color=color, markersize=10 if reference else 6, capsize=3)
                axis.scatter([index - .07, index, index + .07], value['values'], s=13, color=color, alpha=.55)
                included.append({'condition': 'audiomnist_stft', 'method': row['method'], 'metric': metric, 'role': row['role']})
            if target is not None:
                axis.axhline(target, color='gray', linestyle='--', linewidth=1)
            axis.set_ylabel(label)
            labels(axis, names)
        finish(fig, 'audio_content_energy_and_tails', 'AudioMNIST content and energy calibration', included,
               'Complete measured seed triplets, including the verified fixed-spherical plus empirical-gain checkpoint reference. Accuracy is 0.8066667 with population SD 0.00735225, not the historical paper value 0.810 ± 0.013. Dashed coverage targets are 0.05 and 0.01. The reference is neither new arm A nor a matched training-runtime observation. The benchmark constructs gain independently of digit/content; radius-conditioning gains are not assumed.')

    # Dependence-sensitive angular fit in each radius bin.
    rows = [c for c in report['conditions'] if any('angular_sw_bin0' in g['aggregate'] for g in c['methods'].values())]
    if rows:
        fig, axes = plt.subplots(2, 2, figsize=(max(15, len(rows) * .62), 10), squeeze=False)
        included = []
        for bin_index, axis in enumerate(axes.flat):
            metric = f'angular_sw_bin{bin_index}'
            for index, condition in enumerate(rows):
                for offset, (method, group) in zip((-.24, -.08, .08, .24), condition['methods'].items()):
                    value = group['aggregate'].get(metric)
                    if value is None:
                        continue
                    axis.errorbar(index + offset, value['mean'], yerr=value['std_population'], fmt='o', color=colors[method], capsize=2, markersize=3)
                    included.append({'condition': condition['condition_id'], 'method': method, 'metric': metric})
            axis.set_ylabel('Angular sliced W1')
            axis.set_title(f'Test-radius quartile bin {bin_index}; lower is better')
            axis.set_yscale('symlog', linthresh=1e-5)
            labels(axis, [c['condition_id'] for c in rows])
        axes[0, 0].legend(handles=[Line2D([], [], marker='o', linestyle='', color=color, label=method) for method, color in colors.items()], ncol=4)
        finish(fig, 'angular_fit_by_radius_bin', 'Angular fit conditional on radius', included,
               'Each panel uses the existing angular SW evaluator within one test-radius quantile bin. Groups with undefined or nonfinite bins have no aggregate and are omitted explicitly; values are not averaged over surviving seeds.')

    # Radius drift is a numerical invariant diagnostic of A/B/C, not of t-Flow.
    rows = [c for c in report['conditions'] if any(g['radius_drift_summary'].get('mean_relative') for arm, g in c['methods'].items() if arm != 'tflow')]
    if rows:
        fig, axes = plt.subplots(2, 1, figsize=(max(11, len(rows) * .58), 9), squeeze=False)
        included = []
        for axis, metric in zip(axes[:, 0], ('mean_relative', 'max_relative')):
            for index, condition in enumerate(rows):
                for offset, arm in zip((-.2, 0, .2), ('A', 'B', 'C')):
                    value = condition['methods'][arm]['radius_drift_summary'].get(metric)
                    if value is None:
                        continue
                    axis.errorbar(index + offset, value['mean'], yerr=value['std_population'], fmt='o', color=colors[arm], capsize=2)
                    included.append({'condition': condition['condition_id'], 'method': arm, 'radius_drift': metric})
            axis.set_ylabel(metric.replace('_', ' ') + ' radius drift')
            axis.set_yscale('symlog', linthresh=1e-7)
            labels(axis, [c['condition_id'] for c in rows])
        axes[0, 0].legend(handles=[Line2D([], [], marker='o', linestyle='', color=colors[arm], label=arm) for arm in ('A', 'B', 'C')], ncol=3)
        finish(fig, 'ambient_sampler_radius_drift', 'Measured ambient RAFM sampler radius drift', included,
               'Mean and maximum relative drift measured within each seed, summarized over complete seed triplets. t-Flow is excluded from this spherical invariant diagnostic because its population transport intentionally changes radii.')

    rows = [c for c in report['conditions'] if any(g['status'] == 'complete' for g in c['methods'].values())]
    if rows:
        fig, axes = plt.subplots(3, 1, figsize=(max(11, len(rows) * .58), 12), squeeze=False)
        included = []
        for axis, metric, label in zip(axes[:, 0], ('total_train_time_s', 'sample_time_s', 'conditioning_overhead_fraction'), ('Measured train seconds', 'Measured sampling seconds', 'Added parameters / original parameters')):
            for index, condition in enumerate(rows):
                for offset, (method, group) in zip((-.24, -.08, .08, .24), condition['methods'].items()):
                    source = group['parameter_summary'] if metric == 'conditioning_overhead_fraction' else group['aggregate']
                    value = source.get(metric)
                    if value is None:
                        continue
                    axis.errorbar(index + offset, value['mean'], yerr=value['std_population'], fmt='o', color=colors[method], capsize=2, markersize=4)
                    included.append({'condition': condition['condition_id'], 'method': method, 'metric': metric})
            axis.set_ylabel(label)
            if metric != 'conditioning_overhead_fraction':
                axis.set_yscale('symlog', linthresh=1)
            labels(axis, [c['condition_id'] for c in rows])
        axes[0, 0].legend(handles=[Line2D([], [], marker='o', linestyle='', color=color, label=method) for method, color in colors.items()], ncol=4)
        finish(fig, 'measured_runtime_and_parameter_overhead', 'Recorded cost and conditioning overhead', included,
               'Measured current training and sampling wall times include the recorded instrumentation and checkpoint scopes. Hardware/precision differences are retained in report.json; these panels alone do not establish hardware-matched speedups. Missing overhead values are not treated as zero.')
    manifest['status'] = 'measured_figures_exported'
    (output / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    (output / 'figure_captions.md').write_text('\n\n'.join('**' + item['name'] + '.** ' + item['caption'] for item in manifest['figures']) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = plot_report(json.loads(args.report.read_text()), args.output)
    print(json.dumps({'status': manifest['status'], 'figures': len(manifest['figures'])}))


if __name__ == '__main__':
    main()
