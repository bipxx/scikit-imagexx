"""Fixed Montage row-owned campaign calculations; no numerical API calls."""
import math
import statistics
from datetime import datetime


def compare(reports, contract, completed_utc):
    order = contract['measurement']['order']
    assert order == ['A', 'B', 'B', 'A', 'B', 'A', 'A', 'B'] and len(reports) == 8
    expected = contract['case_ids']
    assert all([r['case_id'] for r in report['results']] == expected for report in reports)
    indexed = [{r['case_id']: r for r in report['results']} for report in reports]
    a_indices = [i for i, label in enumerate(order) if label == 'A']
    b_indices = [i for i, label in enumerate(order) if label == 'B']
    gate = contract['gate']
    low, high = contract['measurement_variance_rule']['null_ratio_interval']
    assert low == .97 and high == 1 / .97
    gm = lambda values: math.exp(statistics.mean(math.log(v) for v in values))
    rows = []
    for case_id in expected:
        values = [report[case_id] for report in indexed]
        cpp = [r['cpp_ms'] for r in values]
        reference = [r['reference_ms'] for r in values]
        speed = [r['speedup'] for r in values]
        baseline_cpp = gm(cpp[i] for i in a_indices)
        baseline_normalized = gm(speed[i] for i in a_indices)
        nulls = [dict(passes=[a, b], product=order[a - 1],
                      direct=cpp[a - 1] / cpp[b - 1],
                      reference=reference[a - 1] / reference[b - 1],
                      normalized=speed[b - 1] / speed[a - 1])
                 for a, b in contract['measurement']['same_product_null_pairs']]
        worst_direct = min(cpp[i] for i in a_indices) / max(cpp[i] for i in b_indices)
        worst_normalized = min(speed[i] for i in b_indices) / max(speed[i] for i in a_indices)
        control = case_id != contract['target_case']
        direct_variable = any(not low <= pair['direct'] <= high for pair in nulls)
        normalized_variable = any(not low <= pair['normalized'] <= high for pair in nulls)
        rows.append(dict(case_id=case_id, dtype=values[0]['dtype'], layout=values[0]['layout'],
            reference_ms=reference, cpp_ms=cpp, original_over_cpp=speed,
            baseline_cpp_GM=baseline_cpp, baseline_normalized_GM=baseline_normalized,
            direct_gain=baseline_cpp / gm(cpp[i] for i in b_indices),
            normalized_gain=gm(speed[i] for i in b_indices) / baseline_normalized,
            candidate_original_GM=gm(speed[i] for i in b_indices),
            candidate_direct_passes=[baseline_cpp / cpp[i] for i in b_indices],
            candidate_normalized_passes=[speed[i] / baseline_normalized for i in b_indices],
            candidate_original_passes=[speed[i] for i in b_indices],
            added_ms=gm(cpp[i] for i in b_indices) - baseline_cpp,
            null_pairs=nulls, is_control=control,
            direct_null_outside_band=direct_variable,
            normalized_null_outside_band=normalized_variable,
            worst_direct_gain=worst_direct, worst_normalized_gain=worst_normalized,
            direct_variance_separation_pass=not control or not direct_variable or worst_direct >= .97,
            normalized_variance_separation_pass=not control or not normalized_variable or worst_normalized >= .97))
    controls = [r for r in rows if r['is_control']]
    target = next(r for r in rows if r['case_id'] == contract['target_case'])
    rgb = next(r for r in rows if r['case_id'] == 'rgb-first-f32')
    preserved = [r for r in rows if r['case_id'] in contract['preserved_original_wins']]
    assert len(controls) == 8 and len(preserved) == 6
    for row in rows:
        row['individual_gate_facts'] = dict(
            balanced_direct_floor=not row['is_control'] or row['direct_gain'] >= gate['each_other_eight_balanced_direct_gte'],
            balanced_normalized_floor=not row['is_control'] or row['normalized_gain'] >= gate['each_other_eight_balanced_normalized_gte'],
            preserved_original_passes=[v > gate['each_preserved_original_win_each_B_pass_gt'] for v in row['candidate_original_passes']] if row['case_id'] in contract['preserved_original_wins'] else None,
            target_direct_passes=[v > gate['target_each_B_pass_vs_balanced_A_direct_gt'] for v in row['candidate_direct_passes']] if row is target else None,
            target_normalized_passes=[v > gate['target_each_B_pass_vs_balanced_A_normalized_gt'] for v in row['candidate_normalized_passes']] if row is target else None,
            target_original_passes=[v > gate['target_each_B_pass_original_gt'] for v in row['candidate_original_passes']] if row is target else None,
            rgb_direct_passes=[v >= gate['rgb_first_each_B_pass_direct_gte'] for v in row['candidate_direct_passes']] if row is rgb else None,
            rgb_normalized_passes=[v >= gate['rgb_first_each_B_pass_normalized_gte'] for v in row['candidate_normalized_passes']] if row is rgb else None)
    direct = gm(r['direct_gain'] for r in rows)
    normalized = gm(r['normalized_gain'] for r in rows)
    fixed = dict(
        overall_direct=direct > gate['overall_direct_GM_gt'],
        overall_normalized=normalized > gate['overall_normalized_GM_gt'],
        target_each_candidate_direct=all(target['individual_gate_facts']['target_direct_passes']),
        target_each_candidate_normalized=all(target['individual_gate_facts']['target_normalized_passes']),
        target_each_candidate_original=all(target['individual_gate_facts']['target_original_passes']),
        controls_balanced_direct=all(r['individual_gate_facts']['balanced_direct_floor'] for r in controls),
        controls_balanced_normalized=all(r['individual_gate_facts']['balanced_normalized_floor'] for r in controls),
        preserved_original_each_candidate=all(v for r in preserved for v in r['individual_gate_facts']['preserved_original_passes']),
        rgb_first_each_candidate_direct=all(rgb['individual_gate_facts']['rgb_direct_passes']),
        rgb_first_each_candidate_normalized=all(rgb['individual_gate_facts']['rgb_normalized_passes']))
    deadline_pass = datetime.fromisoformat(completed_utc) < datetime.fromisoformat(contract['deadline_utc'])
    separation_pass = all(r['direct_variance_separation_pass'] and r['normalized_variance_separation_pass'] for r in controls)
    fixed_pass = all(fixed.values())
    status = ('DISCARD' if not deadline_pass else 'UNRESOLVED_MEASUREMENT' if not separation_pass
              else 'KEEP' if fixed_pass else 'DISCARD')
    return dict(status=status, completed_utc=completed_utc, deadline_utc=contract['deadline_utc'],
        completed_before_deadline=deadline_pass, fixed_gate_facts=fixed,
        all_fixed_performance_gates_pass=fixed_pass, metric_variance_separation_pass=separation_pass,
        raw_samples=1008, order=order, candidate_pass_indices=[i + 1 for i in b_indices],
        direct_GM=direct, normalized_GM=normalized,
        candidate_original_GM=gm(r['candidate_original_GM'] for r in rows),
        direct_losses=[r['case_id'] for r in rows if r['direct_gain'] < 1],
        normalized_losses=[r['case_id'] for r in rows if r['normalized_gain'] < 1],
        original_losses=[r['case_id'] for r in rows if r['candidate_original_GM'] < 1],
        unresolved_metrics=[dict(case_id=r['case_id'], metric=metric) for r in controls
                            for metric in ('direct', 'normalized') if not r[metric + '_variance_separation_pass']],
        source_identity_parity_required='All producer/source/wheel/SDK/object/runtime/journal/oracle checks must pass before this function is called.',
        interpretation='One sealed row-owned candidate only. Publish all null variability and worst pairs; a separated gain does not imply measurement stability. Closed experiments remain unchanged.',
        rows=rows)
