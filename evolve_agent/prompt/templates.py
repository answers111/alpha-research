"""
Prompt templates for EvolveAgent
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
"""

# Specialized system template for kissing number problem
KISSING_NUMBER_SYSTEM_TEMPLATE = """You are a computational geometry and optimization specialist working on the kissing number problem in high-dimensional spaces.

Your objectives:
- Understand geometric constraints and sphere packing theory
- Optimize configurations in 11-dimensional Euclidean space
- Ensure mathematical validity while maximizing the number of spheres
- Balance numerical precision with computational feasibility

Your goal is to develop algorithms that find the maximum number of non-overlapping unit spheres that can simultaneously touch a central unit sphere in 11-dimensional space. Focus on maximizing the sphere count while maintaining validity.
"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# Specialized evaluator template for kissing number problem
KISSING_NUMBER_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert in computational geometry and mathematical optimization, specifically evaluating algorithms for the kissing number problem.

Your evaluation focuses on:
- Geometric constraint satisfaction and mathematical correctness
- Sphere configuration quality and optimality
- Numerical precision and stability
- Algorithm efficiency for high-dimensional optimization
"""

# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# Specialized user template for kissing number problem
KISSING_NUMBER_DIFF_USER_TEMPLATE = """# Kissing Number Problem - 11-Dimensional Sphere Packing

## Current Algorithm Performance
- Number of valid kissing spheres: {sphere_count}
- Constraint satisfaction rate: {constraint_satisfaction}
- Primary goal: Maximize sphere count while satisfying constraints

## Mathematical Constraints
1. Non-degeneracy: 0 ∉ C (origin not in sphere centers)
2. Kissing condition: min_{{x≠y∈C}} ||x-y|| ≥ max_{{x∈C}} ||x||

## Areas for Algorithm Improvement
{improvement_areas}

{artifacts}

## Evolution History
{evolution_history}

## Current Algorithm
```{language}
{current_program}
```

## Task: Suggest Algorithm Improvements

Propose targeted changes to improve the sphere count, ensuring all geometric constraints are satisfied. Use the SEARCH/REPLACE format for code suggestions:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code with mathematical reasoning
>>>>>>> REPLACE

Each change should be justified with clear reasoning. Focus on valid and effective ways to increase the number of spheres.
"""

# Specialized full rewrite template for kissing number problem
KISSING_NUMBER_FULL_REWRITE_TEMPLATE = """# Kissing Number Problem - Complete Algorithm Redesign

## Current Performance Analysis
- Current sphere count: {sphere_count}
- Constraint satisfaction: {constraint_satisfaction}
- Key limitations: {improvement_areas}

{artifacts}

## Mathematical Problem Definition
Find the maximum number of non-overlapping unit spheres that can simultaneously touch a central unit sphere in 11-dimensional Euclidean space.

**Constraints:**
1. Non-degeneracy: 0 ∉ C (origin not in the set of sphere centers)
2. Kissing condition: min_{{x≠y∈C}} ||x-y|| ≥ max_{{x∈C}} ||x||

## Evolution History
{evolution_history}

## Current Algorithm (For Reference)
```{language}
{current_program}
```

## Task: Redesign the Algorithm

Develop a new algorithm to improve the sphere count, ensuring all constraints are satisfied. You may use any mathematical or computational approach you find suitable.

**Requirements:**
1. Input: Accept/generate sphere center coordinates in R^11
2. Output: Return valid sphere centers that maximize the count
3. Verification: Include constraint checking functions
4. Optimization: Implement effective search or construction strategies

Provide your complete new algorithm:

```{language}
# Your redesigned kissing number algorithm here
# Include comments explaining the geometric reasoning
```

Your algorithm should be self-contained and ready to run, with proper input/output handling and constraint verification.
"""

DIFF_USER_TEMPLATE_PROPOSAL = """# Previous Proposal: 
{parent_proposal_text}

# Previous Program:
```{language}
{parent_program}
```

# Previous Performance Metrics: 
{metrics}

# Areas Identified for Improvement: 
{improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Proposal
{current_proposal_text}

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""



# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""

# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}
"""

# Template for formatting a previous attempt
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}
"""

# Template for formatting a top program
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}
"""

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
"""

# Specialized evaluation template for kissing number problem
KISSING_NUMBER_EVALUATION_TEMPLATE = """Evaluate the following kissing number algorithm on a scale of 0.0 to 1.0 for the following metrics:

1. **Constraint Satisfaction**: How well does the algorithm ensure all geometric constraints are met?
   - Non-degeneracy condition (origin not in sphere centers)
   - Kissing condition (proper sphere spacing)
   - Mathematical validity of the configuration

2. **Sphere Count Performance**: How effectively does the algorithm maximize the number of spheres?
   - Current sphere count - THIS IS THE ONLY METRIC THAT MATTERS
   - Can it produce MORE spheres than before? 
   - IGNORE everything else - efficiency, elegance, speed are irrelevant

3. **Numerical Stability**: How robust is the algorithm for high-dimensional calculations?
   - Precision handling in 11-dimensional space
   - Stability under floating-point operations
   - Reproducibility and deterministic behavior

4. **Search Effectiveness**: How well does the algorithm find more spheres?
   - Ability to discover configurations with higher sphere counts
   - Success in finding novel sphere arrangements
   - Progress toward breaking current sphere count records

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Algorithm to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "constraint_satisfaction": [score],
    "sphere_count_performance": [score], 
    "numerical_stability": [score],
    "search_effectiveness": [score],
    "reasoning": "[detailed explanation focusing on sphere count maximization]",
    "suggested_improvements": "[specific suggestions for getting MORE SPHERES]"
}}

Focus your evaluation on the mathematical soundness and geometric optimization capabilities rather than general code quality.
"""

# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    # "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE_PROPOSAL,
    # Kissing number problem specialized templates
    "kissing_number_system": KISSING_NUMBER_SYSTEM_TEMPLATE,
    "kissing_number_evaluator_system": KISSING_NUMBER_EVALUATOR_SYSTEM_TEMPLATE,
    "kissing_number_diff_user": KISSING_NUMBER_DIFF_USER_TEMPLATE,
    "kissing_number_full_rewrite": KISSING_NUMBER_FULL_REWRITE_TEMPLATE,
    "kissing_number_evaluation": KISSING_NUMBER_EVALUATION_TEMPLATE
}


class TemplateManager:
    """Manages templates for prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()

        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template
