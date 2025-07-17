#!/usr/bin/env python3
"""
Setup script for kissing number problem using AlphaResearchStarter.
This script generates the initial files needed for the kissing number evolution.
"""

import os
import sys
import asyncio

# Add the parent directory to the path so we can import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from start_alpha_research import AlphaResearchStarter


async def main():
    """Setup the kissing number problem files."""
    
    print("🔢 Setting up 11-Dimensional Kissing Number Problem")
    print("=" * 60)
    
    # Get the current directory (kissing_number)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the research idea for kissing number
    kissing_number_idea = """
    Advanced Algorithmic Approaches for the 11-Dimensional Kissing Number Problem:
    
    Develop computational methods to discover sphere configurations in 11-dimensional 
    Euclidean space that maximize the number of non-overlapping unit spheres that can 
    simultaneously touch a central unit sphere. The current best known lower bound is 
    593 spheres. 
    
    Key constraints:
    1. Non-degeneracy: The origin must not be in the sphere center set
    2. Kissing condition: minimum pairwise distance ≥ maximum norm of centers
    3. Exact integer arithmetic for verification
    
    Technical approach: Multi-strategy generation including random sampling, simplex 
    constructions, and orthogonal expansions, combined with constraint-aware optimization 
    and rigorous mathematical verification.
    """
    
    # Create starter with output directory set to current kissing_number directory
    starter = AlphaResearchStarter(
        config_path=os.path.join(os.path.dirname(current_dir), "configs", "default_config.yaml"),
        output_dir=current_dir
    )
    
    print(f"Output directory: {current_dir}")
    print(f"Research idea: {kissing_number_idea.strip()}")
    print()
    
    # Check if files already exist
    existing_files = ["initial_proposal.txt", "initial_program.py", "evaluator.py"]
    files_exist = any(os.path.exists(os.path.join(current_dir, f)) for f in existing_files)
    
    if files_exist:
        print("⚠️  Some files already exist in this directory:")
        for f in existing_files:
            if os.path.exists(os.path.join(current_dir, f)):
                print(f"   ✓ {f}")
        
        response = input("\nDo you want to overwrite existing files? (y/n): ")
        if response.lower() != 'y':
            print("Aborted. Keeping existing files.")
            return
    
    print("🚀 Generating kissing number problem files...")
    
    try:
        # Generate the files using the workflow (but don't start evolution)
        print("📝 Step 1: Generating research proposal...")
        proposal = await starter.generate_proposal_from_idea(kissing_number_idea)
        print(f"Generated proposal ({len(proposal)} characters)")
        
        print("\n🔍 Step 2: Scoring proposal with reward model...")
        score = await starter.score_proposal(proposal)
        print(f"Proposal score: {score:.2f}/10")
        
        if score < starter.config.rewardmodel.proposal_score_threshold:
            print(f"⚠️  Warning: Proposal score {score:.2f} below threshold {starter.config.rewardmodel.proposal_score_threshold}")
            print("Continuing anyway for kissing number problem setup...")
        
        print("\n💻 Step 3: Generating initial program...")
        # Use our custom initial program instead of generated one
        with open(os.path.join(current_dir, "initial_program.py"), 'r') as f:
            program = f.read()
        print(f"Using existing initial program ({len(program)} characters)")
        
        print("\n🧪 Step 4: Setting up evaluation function...")
        # Use our custom evaluator instead of generated one
        with open(os.path.join(current_dir, "evaluator.py"), 'r') as f:
            evaluator_code = f.read()
        print(f"Using existing evaluator ({len(evaluator_code)} characters)")
        
        print("\n💾 Step 5: Saving files...")
        
        # Save the proposal
        proposal_path = os.path.join(current_dir, "initial_proposal.txt")
        with open(proposal_path, 'w', encoding='utf-8') as f:
            f.write(proposal)
        print(f"Saved proposal to: {proposal_path}")
        
        # Save metadata
        import time
        import json
        metadata = {
            "proposal_score": score,
            "timestamp": time.time(),
            "problem": "11-dimensional kissing number",
            "files": {
                "proposal": proposal_path,
                "program": os.path.join(current_dir, "initial_program.py"),
                "evaluator": os.path.join(current_dir, "evaluator.py")
            }
        }
        metadata_path = os.path.join(current_dir, "generation_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")
        
        print(f"\n{'='*60}")
        print("✅ Setup completed successfully!")
        print(f"Files created in: {current_dir}")
        print("\nGenerated files:")
        print(f"  - initial_proposal.txt (updated)")
        print(f"  - initial_program.py (existing)")
        print(f"  - evaluator.py (existing)")
        print(f"  - generation_metadata.json")
        
        print(f"\nTo start evolution, run:")
        print(f"  python run_kissing_evolution.py")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 