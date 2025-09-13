import math

baselines = {
    "packing_circles_26": {"s_baseline": 2.634, "higher_better": 1},
    "packind_circles_32": {"s_baseline": 2.936, "higher_better": 1},
    "minizing_raio_max_min_distance_d2_n16": {"s_baseline": 12.89, "higher_better": -1},
    "minizing_raio_max_min_distance_d3_n14": {"s_baseline": 4.168, "higher_better": -1},
    "third_autocorrelation_inequality": {"s_baseline": 1.4581, "higher_better": -1},
    # Added benchmarks (larger-is-better where applicable)
    "kissing_number_d11": {"s_baseline": 592.0, "higher_better": 1},
    "spherical_code_d3_n30": {"s_baseline": 0.6736467551690225, "higher_better": 1},
    "heilbronn_in_the_unit_square_n16": {"s_baseline": 7.0/341.0, "higher_better": 1},
    "littlewood_polynomials_n512": {"s_baseline": 0.03125, "higher_better": 1},
    "riesz_energy_n20_s1": {"s_baseline": 0.001013, "higher_better": 1},
    "sum_vs_difference_set_n30": {"s_baseline": 1.04, "higher_better": 1},
    "autoconvolution_peak_minimization": {"s_baseline": 0.6667, "higher_better": 1}
}

results = {
    "packing_circles_26": {"s_best": 2.6359829561164743, "round": 40657},
    "packind_circles_32": {"s_best": 2.939520304932057, "round": 40657},
    "minizing_raio_max_min_distance_d2_n16": {"s_best": 12.92, "round": 5000},
    "minizing_raio_max_min_distance_d3_n14": {"s_best": 5.198, "round": 5000},
    "third_autocorrelation_inequality": {"s_best": 0, "round": 5000},
    # Placeholders for newly added benchmarks (0 indicates not yet attempted)
    "kissing_number_d11": {"s_best": 502.0, "round": 5000},
    "spherical_code_d3_n30": {"s_best": 0.6381359964781541, "round": 5000},
    "heilbronn_in_the_unit_square_n16": {"s_best": 0, "round": 5000},
    "littlewood_polynomials_n512": {"s_best": 0, "round": 5000},
    "riesz_energy_n20_s1": {"s_best": 0, "round": 5000},
    "sum_vs_difference_set_n30": {"s_best": 0, "round": 5000},
    "autoconvolution_peak_minimization": {"s_best": 0, "round": 5000}
}

def compute_excel_best(results):
    problems = list(baselines.keys())
    num_problems = len(problems)
    total = 0.0
    for problem in problems:
        s_baseline = baselines[problem]['s_baseline']
        higher_better = baselines[problem]['higher_better']
        s_best = results[problem]['s_best']
        n_round = results[problem]['round']
        if s_best == 0:
            s_excess = 0  # Assuming s_best == 0 indicates failure/no improvement
        else:
            improvement = (s_best - s_baseline) * higher_better
            s_excess = max(improvement, 0)
        contrib = s_excess / (n_round / 1000000)
        total += contrib
    excel_best = total / num_problems
    return excel_best

print(compute_excel_best(results))