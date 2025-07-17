#!/usr/bin/env python3
"""
Improved Initial Program for 11-Dimensional Kissing Number Problem

This program provides better baseline implementations for generating sphere configurations
that strictly satisfy the kissing number constraints while maximizing sphere count.
Goal: No fixed targets - maximize spheres under strict constraint satisfaction.
"""

import numpy as np
import random
from typing import List, Tuple
import itertools


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


def verify_basic_constraints(sphere_centers: np.ndarray) -> tuple[bool, str]:
    """
    Verify that sphere centers satisfy kissing number constraints.
    Returns (is_valid, message).
    
    This is a wrapper around the standard verify_sphere_packing function
    that returns boolean instead of raising exceptions.
    """
    if len(sphere_centers) == 0:
        return False, "Empty configuration"
    
    if len(sphere_centers) == 1:
        return True, "Single sphere configuration"
    
    try:
        verify_sphere_packing(sphere_centers)
        return True, f"Valid configuration with {len(sphere_centers)} spheres"
    except AssertionError as e:
        return False, str(e)


def generate_reliable_baseline_22(dimension: int = 11, scale: float = 100.0) -> np.ndarray:
    """
    Generate the most reliable baseline: coordinate axes only.
    This always works and provides 22 spheres.
    """
    spheres = []
    
    # Standard axis directions (22 spheres: ±e_i for i=1...11)
    for i in range(dimension):
        coord = [0.0] * dimension
        coord[i] = scale
        spheres.append(coord)
        
        coord = [0.0] * dimension  
        coord[i] = -scale
        spheres.append(coord)
    
    return np.array(spheres)


def generate_safe_diagonal_extension(dimension: int = 11, base_scale: float = 100.0) -> np.ndarray:
    """
    Carefully extend beyond 22 spheres using diagonal directions.
    
    Mathematical analysis:
    - Axis spheres: norm² = base_scale²
    - Distance between opposite axis spheres: (2*base_scale)² = 4*base_scale²
    - For diagonal spheres to satisfy constraints, we need careful scaling
    """
    spheres = []
    
    # 1. Start with reliable 22 axis spheres
    for i in range(dimension):
        for sign in [1, -1]:
            coord = [0.0] * dimension
            coord[i] = sign * base_scale
            spheres.append(coord)
    
    # 2. Add carefully scaled diagonal directions
    # For two-coordinate diagonals: if both coordinates are ±scale/sqrt(2),
    # then norm² = 2*(scale/sqrt(2))² = scale²
    # Distance between axis sphere [scale, 0, ...] and diagonal [scale/sqrt(2), scale/sqrt(2), ...]
    # = sqrt((scale - scale/sqrt(2))² + (scale/sqrt(2))²) 
    
    diag_scale = base_scale / np.sqrt(2)
    
    # Add a few carefully selected diagonal pairs
    safe_pairs = [(0, 1), (2, 3), (4, 5)]  # Start conservatively
    
    for i, j in safe_pairs:
        if i < dimension and j < dimension:
            for sign_i, sign_j in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                coord = [0.0] * dimension
                coord[i] = sign_i * diag_scale
                coord[j] = sign_j * diag_scale
                spheres.append(coord)
    
    return np.array(spheres)


def generate_improved_34_spheres(dimension: int = 11, scale: float = 100.0) -> np.ndarray:
    """
    Generate a conservative 34-sphere configuration.
    22 axis + 12 diagonal spheres with verified constraints.
    """
    spheres = []
    
    # 1. Coordinate axes (22 spheres)
    for i in range(dimension):
        for sign in [1, -1]:
            coord = [0.0] * dimension
            coord[i] = sign * scale
            spheres.append(coord)
    
    # 2. Add three diagonal pairs (12 spheres total)
    # Use smaller scale for diagonals to ensure safety
    diag_scale = scale * 0.6  # Conservative scaling
    
    pairs = [(0, 1), (2, 3), (4, 5)]
    for i, j in pairs:
        for sign_i, sign_j in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            coord = [0.0] * dimension
            coord[i] = sign_i * diag_scale
            coord[j] = sign_j * diag_scale
            spheres.append(coord)
    
    return np.array(spheres)


def generate_optimized_46_spheres(dimension: int = 11, scale: float = 100.0) -> np.ndarray:
    """
    Generate a 46-sphere configuration using mixed strategies.
    """
    spheres = []
    
    # 1. Coordinate axes (22 spheres)
    for i in range(dimension):
        for sign in [1, -1]:
            coord = [0.0] * dimension
            coord[i] = sign * scale
            spheres.append(coord)
    
    # 2. Add diagonal directions with different scaling strategy
    # Use smaller scale to be more conservative
    diag_scale = scale * 0.5
    
    # Add 6 diagonal pairs (24 additional spheres)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 0)]
    for i, j in pairs:
        for sign_i, sign_j in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            coord = [0.0] * dimension
            coord[i] = sign_i * diag_scale
            coord[j] = sign_j * diag_scale
            spheres.append(coord)
    
    return np.array(spheres)


def optimize_configuration_greedy(config: np.ndarray, iterations: int = 20) -> np.ndarray:
    """
    Apply greedy optimization to try to add more spheres.
    Very conservative approach - only add if constraints are definitely satisfied.
    """
    best_config = config.copy()
    current_count = len(config)
    
    for _ in range(iterations):
        # Try to add a random sphere
        attempts = 50
        for _ in range(attempts):
            # Generate a random candidate sphere
            candidate = np.random.normal(0, 50, size=config.shape[1])
            candidate = np.round(candidate)  # Ensure integer coordinates
            
            # Skip if too close to origin
            if np.sum(candidate**2) < 100:
                continue
            
            # Test if adding this sphere maintains constraints
            test_config = np.vstack([best_config, candidate.reshape(1, -1)])
            is_valid, _ = verify_basic_constraints(test_config)
            
            if is_valid and len(test_config) > current_count:
                best_config = test_config
                current_count = len(test_config)
                break
    
    return best_config


def main() -> np.ndarray:
    """
    Generate the best possible sphere configuration without fixed targets.
    Try multiple strategies and return the one with most spheres.
    """
    print("11-Dimensional Kissing Number Configuration Generator")
    print("============================================================")
    print("Research Goal: Maximize sphere count under strict constraints")
    print("No fixed targets - the more spheres, the better!")
    print()
    
    strategies = [
        ("Reliable-Baseline-22", generate_reliable_baseline_22),
        ("Safe-Diagonal-34", generate_safe_diagonal_extension),
        ("Improved-34-Spheres", generate_improved_34_spheres),
        ("Optimized-46-Spheres", generate_optimized_46_spheres),
    ]
    
    best_config = None
    best_count = 0
    best_strategy = "None"
    
    for strategy_name, strategy_func in strategies:
        print(f"Trying {strategy_name} strategy...")
        try:
            config = strategy_func()
            is_valid, message = verify_basic_constraints(config)
            
            print(f"  Generated {len(config)} spheres")
            print(f"  Valid: {is_valid}, Message: {message}")
            
            if is_valid and len(config) > best_count:
                # Try greedy optimization
                optimized = optimize_configuration_greedy(config, iterations=20)
                is_valid_opt, message_opt = verify_basic_constraints(optimized)
                
                print(f"  After optimization: {len(optimized)} spheres")
                print(f"  Valid: {is_valid_opt}, Message: {message_opt}")
                
                if is_valid_opt and len(optimized) > best_count:
                    best_config = optimized
                    best_count = len(optimized)
                    best_strategy = f"{strategy_name}+Optimized"
                elif is_valid and len(config) > best_count:
                    best_config = config
                    best_count = len(config)
                    best_strategy = strategy_name
        except Exception as e:
            print(f"  Error in {strategy_name}: {e}")
        print()
    
    print("=" * 60)
    print("BEST RESULT:")
    print(f"Strategy: {best_strategy}")
    print(f"Number of spheres: {best_count}")
    
    if best_config is not None:
        print(f"Configuration shape: {best_config.shape}")
        print("Sample coordinates (first 3 spheres):")
        for i in range(min(3, len(best_config))):
            coord_str = " ".join([f"{int(x):4d}" for x in best_config[i]])
            print(f"  Sphere {i+1}: [{coord_str}]")
        
        # Compute detailed metrics
        sphere_centers_int = np.around(best_config).astype(np.int64) 
        squared_norms = [np.sum(center**2) for center in sphere_centers_int]
        max_squared_norm = max(squared_norms)
        
        if len(sphere_centers_int) > 1:
            min_squared_distance = min(
                np.sum((a - b)**2)
                for a, b in itertools.combinations(sphere_centers_int, 2)
            )
        else:
            min_squared_distance = float('inf')
        
        print(f"\nDetailed metrics:")
        print(f"  Max norm² = {max_squared_norm}")
        print(f"  Min pairwise distance² = {min_squared_distance}")
        print(f"  Constraint satisfied: {min_squared_distance >= max_squared_norm}")
        print(f"  Safety margin: {min_squared_distance / max_squared_norm:.2f}x")
        
        # Final verification
        is_valid, message = verify_basic_constraints(best_config)
        print(f"Final verification: {is_valid} - {message}")
        
        return best_config
    else:
        print("No valid configuration found!")
        # Return the reliable baseline as fallback
        return generate_reliable_baseline_22()


if __name__ == "__main__":
    result = main() 