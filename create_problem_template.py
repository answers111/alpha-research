#!/usr/bin/env python3
"""
Template generator for creating new problem directories with the Alpha Research framework.
"""

import os
import shutil
import argparse
from pathlib import Path


def create_problem_template(problem_name: str, base_dir: str = "."):
    """
    Create a new problem directory based on the kissing_number template.
    
    Args:
        problem_name: Name of the new problem
        base_dir: Base directory to create the problem folder in
    """
    
    # Define paths
    template_dir = os.path.join(base_dir, "kissing_number")
    new_problem_dir = os.path.join(base_dir, problem_name)
    
    # Check if template exists
    if not os.path.exists(template_dir):
        print(f"❌ Template directory not found: {template_dir}")
        return False
    
    # Check if target directory already exists
    if os.path.exists(new_problem_dir):
        print(f"❌ Directory already exists: {new_problem_dir}")
        return False
    
    print(f"🛠️  Creating new problem template: {problem_name}")
    print(f"Source template: {template_dir}")
    print(f"Target directory: {new_problem_dir}")
    print()
    
    try:
        # Create the new directory
        os.makedirs(new_problem_dir, exist_ok=True)
        
        # Files to copy from template
        template_files = [
            "initial_program.py",
            "initial_proposal.txt", 
            "evaluator.py",
            "run_kissing_evolution.py",
            "setup_kissing_evolution.py",
            "test_setup.py",
            "README.md"
        ]
        
        # Copy and modify files
        for file_name in template_files:
            source_path = os.path.join(template_dir, file_name)
            target_path = os.path.join(new_problem_dir, file_name)
            
            if os.path.exists(source_path):
                # Read the source file
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace references to kissing_number with new problem name
                content = content.replace("kissing_number", problem_name)
                content = content.replace("Kissing Number", problem_name.replace("_", " ").title())
                content = content.replace("11-Dimensional Kissing Number", f"{problem_name.replace('_', ' ').title()} Problem")
                content = content.replace("kissing number", f"{problem_name.replace('_', ' ')} problem")
                
                # Write the modified content
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✓ Created {file_name}")
            else:
                print(f"⚠️  Template file not found: {file_name}")
        
        # Create a basic problem description file
        problem_md_content = f"""# {problem_name.replace('_', ' ').title()} Problem

## Problem Statement

[Add your problem description here]

## Objective

[Define the optimization objective]

## Constraints

[List the constraints and requirements]

## Evaluation Criteria

[Describe how solutions will be evaluated]

## Files

- **`initial_program.py`** - Starting program for evolution
- **`initial_proposal.txt`** - Research proposal describing the approach  
- **`evaluator.py`** - Evaluation function for the evolution framework
- **`run_{problem_name}_evolution.py`** - Main script to start evolution
- **`setup_{problem_name}_evolution.py`** - Setup script for generating files
- **`test_setup.py`** - Test script to verify setup

## Quick Start

1. Customize the problem description in this file
2. Modify `initial_program.py` for your problem
3. Update `evaluator.py` with appropriate metrics
4. Run setup: `python setup_{problem_name}_evolution.py`
5. Start evolution: `python run_{problem_name}_evolution.py`

## Notes

This template was generated from the kissing_number problem template.
Please customize all files according to your specific problem requirements.
"""
        
        with open(os.path.join(new_problem_dir, f"{problem_name}.md"), 'w') as f:
            f.write(problem_md_content)
        print(f"✓ Created {problem_name}.md")
        
        print(f"\n✅ Successfully created problem template: {problem_name}")
        print(f"\nNext steps:")
        print(f"1. cd {problem_name}")
        print(f"2. Edit {problem_name}.md to describe your problem")
        print(f"3. Customize initial_program.py for your problem")
        print(f"4. Update evaluator.py with appropriate metrics")
        print(f"5. Run: python setup_{problem_name}_evolution.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating template: {e}")
        # Clean up on error
        if os.path.exists(new_problem_dir):
            shutil.rmtree(new_problem_dir)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create a new problem template for Alpha Research")
    parser.add_argument("problem_name", help="Name of the new problem (e.g., 'traveling_salesman')")
    parser.add_argument("--base-dir", default=".", help="Base directory to create the problem folder in")
    
    args = parser.parse_args()
    
    # Validate problem name
    if not args.problem_name.replace("_", "").isalnum():
        print("❌ Problem name should contain only letters, numbers, and underscores")
        return
    
    success = create_problem_template(args.problem_name, args.base_dir)
    
    if success:
        print(f"\n🎉 Problem template '{args.problem_name}' created successfully!")
    else:
        print(f"\n💥 Failed to create problem template '{args.problem_name}'")


if __name__ == "__main__":
    main() 