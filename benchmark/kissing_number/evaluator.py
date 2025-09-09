#@title Verification
import numpy as np
import subprocess
import sys
import traceback
import os
import json

def verify_kissing_configuration(sphere_centers: np.ndarray, atol: float = 1e-9):
    """
    Verifies if the given points form a valid kissing number configuration.

    A valid kissing configuration of N vectors in D dimensions must satisfy:
    1. All vectors are unit vectors (norm is 1).
    2. The dot product of any two distinct vectors is at most 0.5.

    Args:
      sphere_centers: A numpy array of shape (N, D) where N is the number of spheres
                      and D is the dimension.
      atol: Absolute tolerance for floating point comparisons.

    Raises:
      AssertionError: If the configuration is not valid.
    """
    num_spheres, dimension = sphere_centers.shape
    
    # 1. Check if all vectors are unit vectors.
    norms = np.linalg.norm(sphere_centers, axis=1)
    assert np.allclose(norms, 1.0, atol=atol), f"Verification failed: Not all vectors are unit vectors. Norms range from {np.min(norms)} to {np.max(norms)}."
    
    # 2. Check the dot products.
    # The dot product of two distinct vectors must be <= 0.5.
    dot_products = sphere_centers @ sphere_centers.T
    
    # We only need to check the upper triangle, excluding the diagonal.
    # The diagonal elements should be 1.0 for unit vectors.
    np.fill_diagonal(dot_products, -np.inf) # so we don't pick diagonal elements
    
    max_dot_product = np.max(dot_products)
    
    # The condition is dot_product <= 0.5
    assert max_dot_product <= 0.5 + atol, f"Verification failed: Maximum dot product between distinct vectors is {max_dot_product}, which is greater than 0.5."


def evaluate(program_path: str):
  """
  Evaluate a program that solves the kissing number problem.
  
  Args:
    program_path: Path to the Python program file to evaluate
    
  Returns:
    dict: Dictionary with metric names as keys and numeric scores as values
  """
  try:
    # Use importlib.util to dynamically load the program module
    import importlib.util
    
    # Load the module from the given path
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    sys.modules["program"] = program
    spec.loader.exec_module(program)
    
    # Look for sphere_centers in the loaded module
    sphere_centers = None
    if hasattr(program, 'sphere_centers'):
      sphere_centers = program.sphere_centers
    elif hasattr(program, 'main'):
      # If there's a main function, try calling it to get sphere_centers
      sphere_centers = program.main()
    
    if sphere_centers is None:
      return {"error": -1.0, "no_sphere_centers": True}
    
    # Verify the kissing configuration
    verify_kissing_configuration(sphere_centers)
    
    # Calculate metrics
    num_spheres = sphere_centers.shape[0]
    dimension = sphere_centers.shape[1]
    
    # Return metrics with proper string keys
    return {
      "num_spheres": float(num_spheres),
      "dimension": float(dimension)
    }
    
  except Exception as e:
    return {"error": -1.0, "evaluation_error": True, "stderr": traceback.format_exc()}


if __name__ == "__main__":
  # CLI for debugging: evaluate initial_program.py by default, or a provided path
  try:
    default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
  except Exception:
    default_path = "initial_program.py"

  target = sys.argv[1] if len(sys.argv) > 1 else default_path
  print(json.dumps(evaluate(target), ensure_ascii=False, indent=2))