import math

baselines = {
    "packing_circles_26": {"s_baseline": 2.634, "higher_better": 1},
    "packind_circles_32": {"s_baseline": 2.936, "higher_better": 1},
    "minizing_raio_max_min_distance_d2_n16": {"s_baseline": 12.89, "higher_better": -1},
    "minizing_raio_max_min_distance_d3_n14": {"s_baseline": 4.168, "higher_better": -1},
    "third_autocorrelation_inequality": {"s_baseline": 1.4581, "higher_better": -1}
}

results = {
    "packing_circles_26": {"s_best": 2.6359829561164743, "round": 40657},
    "packind_circles_32": {"s_best": 2.939520304932057, "round": 40657},
    "minizing_raio_max_min_distance_d2_n16": {"s_best": 12.92, "round": 5000},
    "minizing_raio_max_min_distance_d3_n14": {"s_best": 5.198, "round": 5000},
    "third_autocorrelation_inequality": {"s_best": 0, "round": 5000},
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