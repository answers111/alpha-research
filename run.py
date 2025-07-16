from evolve_agent import EvolveAgent    
import asyncio
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# logging.basicConfig(
#     level=logging.DEBUG, 
#     # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
#     # filename='app.log',  
# )

evolve_agent = EvolveAgent(
    initial_program_path="kissing_number/initial_program.py",
    evaluation_file="kissing_number/evaluator.py",
    initial_proposal_path="kissing_number/initial_proposal.txt",
    config_path="configs/default_config.yaml",
    output_dir="kissing_number/evolve_agent_output"
)

async def main():
    best_program = await evolve_agent.run(iterations=50) 
    print(best_program)

asyncio.run(main())
# print(evolve_agent)

