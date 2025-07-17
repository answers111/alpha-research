import numpy as np
import itertools
import subprocess
import sys
import os
import importlib.util


def compute_squared_norm(point: list[int]) -> int:
    """Returns the squared norm of an integer vector using exact computation."""
    return sum(pow(int(x), 2) for x in point)


def verify_sphere_packing(sphere_centers: np.ndarray):
    """Checks that after normalizing, the points correspond to a valid sphere packing for kissing numbers.

    Args:
        sphere_centers: the list of sphere centers, of shape [num_spheres, dimension].

    Raises:
        AssertionError: if the sphere packing is not a valid kissing configuration.
    """
    # Rounding to integers to guarantee exact computation throughout.
    sphere_centers = np.around(sphere_centers).astype(np.int64)
    squared_norms = [compute_squared_norm(list(center)) for center in sphere_centers]

    # Checks that the set doesn't contain 0.
    min_squared_norm = min(squared_norms)
    assert min_squared_norm > 1e-6, f'Verification failed because the set contains 0.'

    # Checks that the minimum pairwise distance between centers >= the maximum norm of the centers.
    max_squared_norm = max(squared_norms)
    min_squared_distance = min(compute_squared_norm(list(a - b)) for a, b in itertools.combinations(sphere_centers, 2))
    assert min_squared_distance >= max_squared_norm, f'Verification failed because the minimum squared distance = {min_squared_distance} < {max_squared_norm} = maximum squared norm.'


def evaluate(program_path: str) -> dict:
    """
    Evaluate a kissing number program and return performance metrics.
    
    Research Goal: Maximize sphere count while strictly satisfying constraints.
    No fixed targets - the more spheres, the better.
    
    Args:
        program_path: Path to the Python program file to evaluate
        
    Returns:
        dict: Dictionary with metric names as keys and numeric scores as values
    """
    metrics = {}
    
    try:
        # Import the program module
        spec = importlib.util.spec_from_file_location("program_module", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load program from {program_path}")
        
        program_module = importlib.util.module_from_spec(spec)
        sys.modules["program_module"] = program_module
        spec.loader.exec_module(program_module)
        
        # Execute the program and capture the sphere configuration
        if hasattr(program_module, 'main'):
            # Redirect stdout to capture output
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            try:
                sphere_centers = program_module.main()
                output = mystdout.getvalue()
            finally:
                sys.stdout = old_stdout
            
            if sphere_centers is not None and len(sphere_centers) > 0:
                # Use the standard verification function
                try:
                    num_spheres = sphere_centers.shape[0]
                    dimension = sphere_centers.shape[1]
                    
                    # CRITICAL: Check dimension FIRST before any scoring
                    if dimension != 11:
                        print(f'Verification failed: Wrong dimension {dimension}, expected 11.')
                        # Dimension is wrong - all metrics are zero
                        metrics["valid_configuration"] = 0.0
                        metrics["num_spheres"] = 0.0
                        metrics["fitness_score"] = 0.0
                        metrics["combined_score"] = 0.0  # 0 spheres = lowest rank
                        metrics["constraint_margin"] = 0.0
                        metrics["configuration_efficiency"] = 0.0
                        metrics["correct_dimension"] = 0.0
                    else:
                        # Dimension is correct, proceed with verification
                        verify_sphere_packing(sphere_centers)
                        print(f'Verified the sphere packing showing kissing number in dimension {dimension} is at least {num_spheres}.')
                        
                        # ===== PURE SPHERE COUNT OPTIMIZATION =====
                        # Primary objective: maximize sphere count (no upper bound)
                        metrics["num_spheres"] = float(num_spheres)
                        
                        # Main fitness score: EXACTLY equal to sphere count for pure comparison
                        metrics["fitness_score"] = float(num_spheres)
                        
                        # Combined score: EXACTLY equal to sphere count for pure ranking
                        # This ensures that only sphere count determines program ranking
                        metrics["combined_score"] = float(num_spheres)
                        
                        # Constraint satisfaction: must be perfect (1.0) for valid solutions
                        metrics["valid_configuration"] = 1.0
                        
                        # Dimension correctness: binary check
                        metrics["correct_dimension"] = 1.0
                        
                        # Quality metrics (for analysis only, NOT used in comparison)
                        sphere_centers_int = np.around(sphere_centers).astype(np.int64)
                        squared_norms = [compute_squared_norm(list(center)) for center in sphere_centers_int]
                        
                        if len(squared_norms) > 1:
                            max_squared_norm = max(squared_norms)
                            min_squared_distance = min(
                                compute_squared_norm(list(a - b)) 
                                for a, b in itertools.combinations(sphere_centers_int, 2)
                            )
                            
                            # Constraint margin: how much "safety margin" we have
                            constraint_margin = min_squared_distance / max_squared_norm if max_squared_norm > 0 else 1.0
                            metrics["constraint_margin"] = min(10.0, constraint_margin)  # Cap at 10x margin
                            
                            # Configuration efficiency: compactness of the solution
                            avg_norm_squared = sum(squared_norms) / len(squared_norms)
                            if avg_norm_squared > 0:
                                metrics["configuration_efficiency"] = 1.0 / (1.0 + avg_norm_squared / 10000.0)
                            else:
                                metrics["configuration_efficiency"] = 1.0
                        else:
                            metrics["constraint_margin"] = 1.0
                            metrics["configuration_efficiency"] = 1.0
                    
                except AssertionError as e:
                    print(f'Verification failed: {e}')
                    # For invalid configurations, all metrics are zero except dimension check
                    # Combined score = 0.0 (no spheres) ensures proper ranking
                    metrics["valid_configuration"] = 0.0
                    metrics["num_spheres"] = 0.0
                    metrics["fitness_score"] = 0.0
                    metrics["combined_score"] = 0.0  # 0 spheres = lowest rank
                    metrics["constraint_margin"] = 0.0
                    metrics["configuration_efficiency"] = 0.0
                    metrics["correct_dimension"] = 1.0 if sphere_centers.shape[1] == 11 else 0.0
                    
            else:
                print("Program did not return a valid sphere configuration")
                metrics["valid_configuration"] = 0.0
                metrics["num_spheres"] = 0.0
                metrics["fitness_score"] = 0.0
                metrics["combined_score"] = 0.0  # 0 spheres = lowest rank
                metrics["constraint_margin"] = 0.0
                metrics["configuration_efficiency"] = 0.0
                metrics["correct_dimension"] = 0.0
        else:
            print("Program does not have a main() function")
            metrics["execution_success"] = 0.0
            metrics["num_spheres"] = 0.0
            metrics["valid_configuration"] = 0.0
            metrics["fitness_score"] = 0.0
            metrics["combined_score"] = 0.0  # 0 spheres = lowest rank
            
    except Exception as e:
        print(f"Error executing program: {e}")
        metrics["execution_success"] = 0.0
        metrics["error"] = 0.0
        metrics["num_spheres"] = 0.0
        metrics["valid_configuration"] = 0.0
        metrics["fitness_score"] = 0.0
        metrics["combined_score"] = 0.0  # 0 spheres = lowest rank
        
    return metrics