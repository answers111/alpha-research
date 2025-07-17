#!/usr/bin/env python3
"""
Script to run the EvolveAgent on the 11-dimensional kissing number problem.
"""

import os
import sys
import asyncio

# Add the parent directory to the path so we can import from evolve_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolve_agent import EvolveAgent


async def main():
    """Run the kissing number evolution."""
    
    # Get the current directory (kissing_number)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    initial_program_path = os.path.join(current_dir, "initial_program.py")
    initial_proposal_path = os.path.join(current_dir, "initial_proposal.txt")
    evaluator_path = os.path.join(current_dir, "evaluator.py")
    config_path = os.path.join(os.path.dirname(current_dir), "configs", "default_config.yaml")
    
    # Check that all required files exist
    required_files = [initial_program_path, initial_proposal_path, evaluator_path, config_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return
    
    print("11-Dimensional Kissing Number Evolution")
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
            output_dir=os.path.join(current_dir, "evolve_agent_output")
        )
        
        # Configure specialized templates for kissing number problem
        print("Configuring kissing number specialized templates...")
        print("  - System: Expert in computational geometry and sphere packing")
        print("  - User prompts: Focus on 11D constraints and sphere maximization")  
        print("  - Evaluation: Geometric correctness and optimization metrics")
        
        evolve_agent.prompt_sampler.set_templates(
            system_template="kissing_number_system",
            user_template="kissing_number_diff_user"
        )
        evolve_agent.evaluator_prompt_sampler.set_templates(
            system_template="kissing_number_evaluator_system",
            user_template="kissing_number_evaluation"
        )
        
        print("Templates configured for 11-dimensional kissing number optimization")
        print("Note: Will automatically use 'kissing_number_full_rewrite' for full rewrites")
        print("Starting evolution...")
        
        # Run the evolution for a specified number of iterations
        best_program = await evolve_agent.run(iterations=100)
        
        if best_program:
            print("\nEvolution completed successfully!")
            print(f"Best program ID: {best_program.id}")
            print(f"Best metrics: {best_program.metrics}")
            print(f"Number of spheres: {best_program.metrics.get('num_spheres', 0)}")
            
            # Save the best program
            best_program_path = os.path.join(current_dir, "best_kissing_program.py")
            with open(best_program_path, 'w') as f:
                f.write(best_program.code)
            print(f"Best program saved to: {best_program_path}")
            
        else:
            print("Evolution completed but no valid programs found.")
            
    except Exception as e:
        print(f"Error during evolution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set CUDA device if available
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    
    asyncio.run(main()) 