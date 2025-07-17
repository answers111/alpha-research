#!/usr/bin/env python3
"""
Test script to verify the kissing number setup is working correctly.
"""

import os
import sys
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_initial_program():
    """Test the initial program execution."""
    print("Testing initial_program.py...")
    
    try:
        # Import and run the initial program
        from initial_program import main
        
        # Run the main function
        sphere_centers = main()
        
        if sphere_centers is not None:
            print(f"✅ Initial program executed successfully")
            print(f"   Generated {len(sphere_centers)} spheres")
            print(f"   Shape: {sphere_centers.shape}")
            return True
        else:
            print("❌ Initial program returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error running initial program: {e}")
        return False


def test_evaluator():
    """Test the evaluator function."""
    print("\nTesting evaluator.py...")
    
    try:
        from evaluator import evaluate
        
        # Create a test program file
        test_program_content = '''
import numpy as np

def main():
    # Simple test configuration - just a few orthogonal vectors
    points = []
    scale = 1000
    
    # Add some basis vectors
    for i in range(3):
        vec = np.zeros(11)
        vec[i] = scale
        points.append(vec)
    
    return np.array(points, dtype=np.int64)

if __name__ == "__main__":
    result = main()
    print(f"Generated {len(result)} spheres")
'''
        
        # Write test program
        test_program_path = "test_program.py"
        with open(test_program_path, 'w') as f:
            f.write(test_program_content)
        
        # Evaluate the test program
        metrics = evaluate(test_program_path)
        
        print(f"✅ Evaluator executed successfully")
        print(f"   Metrics: {metrics}")
        
        # Clean up
        if os.path.exists(test_program_path):
            os.remove(test_program_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error running evaluator: {e}")
        # Clean up on error
        if os.path.exists("test_program.py"):
            os.remove("test_program.py")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "kissing.md",
        "data.py", 
        "initial_program.py",
        "initial_proposal.txt",
        "evaluator.py",
        "run_kissing_evolution.py"
    ]
    
    all_exist = True
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            all_exist = False
    
    return all_exist


def test_config_compatibility():
    """Test compatibility with the main config."""
    print("\nTesting config compatibility...")
    
    try:
        config_path = "../configs/default_config.yaml"
        if os.path.exists(config_path):
            print(f"✅ Config file found: {config_path}")
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Kissing Number Setup Test")
    print("=" * 40)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Initial Program", test_initial_program),
        ("Evaluator", test_evaluator),
        ("Config Compatibility", test_config_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 20)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*40}")
    print("📊 TEST SUMMARY")
    print(f"{'='*40}")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not success:
            all_passed = False
    
    print(f"\n{'='*40}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Setup is ready for evolution.")
        print("\nTo start evolution, run:")
        print("  python run_kissing_evolution.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running evolution.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 