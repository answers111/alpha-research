#!/usr/bin/env python3
"""
Quick test script for kissing number problem with Alpha Research.
Runs a short evolution to test the system.
"""

import os
import sys
import asyncio

# Add the parent directory to the path so we can import from evolve_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolve_agent import EvolveAgent


async def main():
    """Run a quick test of the kissing number evolution."""
    
    # Get the current directory (kissing_number)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    initial_program_path = os.path.join(current_dir, "initial_program.py")
    initial_proposal_path = os.path.join(current_dir, "initial_proposal.txt")
    evaluator_path = os.path.join(current_dir, "evaluator.py")
    config_path = os.path.join(os.path.dirname(current_dir), "configs", "default_config.yaml")
    
    print("🧪 11-Dimensional Kissing Number Quick Test")
    print("=" * 50)
    print(f"Initial program: {initial_program_path}")
    print(f"Initial proposal: {initial_proposal_path}")
    print(f"Evaluator: {evaluator_path}")
    print(f"Config: {config_path}")
    print()
    
    # Create the EvolveAgent
    try:
        evolve_agent = EvolveAgent(
            initial_program_path=initial_program_path,
            initial_proposal_path=initial_proposal_path,
            evaluation_file=evaluator_path,
            config_path=config_path,
            output_dir=os.path.join(current_dir, "test_output")
        )
        
        print("🚀 Starting quick test evolution (5 iterations)...")
        print("This will test:")
        print("  ✓ Reward model API calls")
        print("  ✓ Proposal generation and scoring")
        print("  ✓ Code evolution")
        print("  ✓ Evaluation function")
        print()
        
        # Run a short evolution for testing
        best_program = await evolve_agent.run(iterations=5)
        
        if best_program:
            print("\n🎉 Quick test completed successfully!")
            print(f"Best program ID: {best_program.id}")
            print(f"Best metrics: {best_program.metrics}")
            
            num_spheres = best_program.metrics.get('num_spheres', 0)
            valid_config = best_program.metrics.get('valid_configuration', 0)
            
            print(f"\n📊 Results Summary:")
            print(f"  Number of spheres: {num_spheres}")
            print(f"  Valid configuration: {'Yes' if valid_config > 0.5 else 'No'}")
            print(f"  Normalized score: {best_program.metrics.get('normalized_score', 0):.4f}")
            
            # Save the test result
            test_program_path = os.path.join(current_dir, "test_best_program.py")
            with open(test_program_path, 'w') as f:
                f.write(best_program.code)
            print(f"  Test program saved to: {test_program_path}")
            
            print(f"\n✅ System is working correctly!")
            print(f"💡 To run full evolution, use: python run_kissing_evolution.py")
            
        else:
            print("❌ Test failed - no valid programs found.")
            print("This might indicate issues with:")
            print("  - API configuration")
            print("  - Proposal scoring")
            print("  - Code generation")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 Troubleshooting suggestions:")
        print(f"  1. Check API key in configs/default_config.yaml")
        print(f"  2. Verify internet connection")
        print(f"  3. Run: python test_setup.py")


if __name__ == "__main__":
    # Set CUDA device if available (though we're using API)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    
    asyncio.run(main()) 