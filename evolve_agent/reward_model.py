import re
import os
import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path

import asyncio
# import aiofiles
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from datasets import load_dataset

from evolve_agent.config import RewardModelConfig


class RewardModel:
    """
    A class to score research abstracts or proposals based on clarity, novelty, technical rigor,
    and potential impact using either a local vLLM model or an external API model.
    """
    
    BOX = r"\boxed{}"
    SYSTEM_PROMPT = "You are an expert reviewer tasked with evaluating the quality of a research proposal."
    SCORING_PROMPT = f"""
Your goal is to assign a score between 1 and 10 based on the proposal's clarity, novelty, technical rigor, and potential impact. Here are the criteria:
1. Read the following proposal carefully and provide a score from 1 to 10. 
2. Score 6 means slightly higher than the borderline, 5 is slightly lower than the borderline.
Write the score in the {BOX}.
**idea**:
"""

    def __init__(self, config: RewardModelConfig):
        """
        Initialize the RewardModel.

        Args:
            config (RewardModelConfig): Configuration object containing model_type, model_name, api_key, base_url,
                                       jsonl_file, max_retries, retry_delay, temperature, top_p, max_tokens.
        """
        self.config = config

        if self.config.model_type == "vllm":
            self.llm = LLM(model=self.config.model_name, gpu_memory_utilization=0.95)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        elif self.config.model_type == "api":
            if not self.config.api_key or not self.config.base_url:
                raise ValueError("API key and base URL must be provided for API model type.")
            self.client = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        else:
            raise ValueError("model_type must be 'vllm' or 'api'.")

        # Ensure the directory for jsonl_file exists
        os.makedirs(os.path.dirname(self.config.jsonl_file) or ".", exist_ok=True)

    def parse_score_from_text(self, text: str) -> float:
        """
        Parse the score from the model's output text.

        Args:
            text (str): Model output containing the score in \boxed{} format.

        Returns:
            float: Parsed score between 0 and 10, or -1.0 if invalid.
        """
        match = re.search(r'\\boxed\{(\d*\.?\d*)\}', text)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                pass
        return -1.0

    async def load_processed_ids(self) -> set:
        """
        Load titles of already processed abstracts from the JSONL file.

        Returns:
            set: Set of processed abstract titles with valid scores.
        """
        processed_ids = set()
        # if os.path.exists(self.config.jsonl_file):
        #     async with aiofiles.open(self.config.jsonl_file, 'r', encoding='utf-8') as f:
        #         async for line in f:
        #             try:
        #                 data = json.loads(line.strip())
        #                 if data.get('score', -1.0) != -1.0:
        #                     processed_ids.add(data['title'])
        #             except json.JSONDecodeError:
        #                 print(f"Warning: Skipping invalid JSON line in {self.config.jsonl_file}")
        return processed_ids

    async def write_result_to_jsonl(self, result: Dict):
        """
        Write a single result to the JSONL file if the score is valid.

        Args:
            result (Dict): Result dictionary containing title, score, evaluation, abstract, and gt_score.
        """
        # if result['score'] != -1.0:
        #     async with aiofiles.open(self.config.jsonl_file, 'a', encoding='utf-8') as f:
        #         await f.write(json.dumps(result, ensure_ascii=False) + '\n')

    async def score_with_vllm(self, data: List[Dict]) -> List[Dict]:
        """
        Score abstracts using a local vLLM model.

        Args:
            data (List[Dict]): List of dictionaries containing 'title', 'abstract', and 'gt_score'.

        Returns:
            List[Dict]: List of results with 'title', 'score', 'evaluation', 'abstract', and 'gt_score'.
        """
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.SCORING_PROMPT + item["title"] + "\n" + item["abstract"]}
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for item in data
        ]

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )

        # vLLM is synchronous, so we run it in the default executor
        outputs = await asyncio.get_event_loop().run_in_executor(None, lambda: self.llm.generate(prompts, sampling_params))

        results = []
        for output, item in zip(outputs, data):
            output_text = output.outputs[0].text.strip()
            score = self.parse_score_from_text(output_text)
            result = {
                "title": item["title"],
                "score": score,
                "evaluation": output_text,
                "abstract": item["abstract"],
                "gt_score": item["gt_score"]
            }
            await self.write_result_to_jsonl(result)
            results.append(result)

        return results

    async def score_with_api(self, data: List[Dict]) -> List[Dict]:
        """
        Score abstracts using an external API model.

        Args:
            data (List[Dict]): List of dictionaries containing 'title', 'abstract', and 'gt_score'.

        Returns:
            List[Dict]: List of results with 'title', 'score', 'evaluation', 'abstract', and 'gt_score'.
        """
        processed_ids = await self.load_processed_ids()
        data_to_process = [item for item in data if item['title'] not in processed_ids]
        print(f"Total abstracts: {len(data)}, To process: {len(data_to_process)}, Already processed: {len(processed_ids)}")

        prompts = [
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.SCORING_PROMPT + item["title"] + "\n" + item["abstract"]}
            ]
            for item in data_to_process
        ]

        results = []
        for prompt, item in zip(prompts, data_to_process):
            retries = 0
            score = -1.0
            output_text = ""

            while score == -1.0 and retries < self.config.max_retries:
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=prompt,
                        temperature=0,  # API uses fixed temperature as per original code
                        max_tokens=1000,  # API uses fixed max_tokens as per original code
                        top_p=1.0  # API uses fixed top_p as per original code
                    )
                    output_text = response.choices[0].message.content.strip()
                    score = self.parse_score_from_text(output_text)
                except Exception as e:
                    print(f"Error processing {item['title']}: {e}")
                
                if score == -1.0:
                    retries += 1
                    print(f"Invalid score for abstract: {item['title']}, Retry {retries}/{self.config.max_retries}")
                    await asyncio.sleep(self.config.retry_delay)

            result = {
                "title": item["title"],
                "score": score,
                "gt_score": item["gt_score"],
                "evaluation": output_text,
                "abstract": item["abstract"]
            }
            await self.write_result_to_jsonl(result)
            results.append(result)

            if score == -1.0:
                print(f"Failed to get valid score for abstract: {item['title']} after {self.config.max_retries} retries")

        # Include previously processed results
        # if processed_ids:
        #     async with aiofiles.open(self.config.jsonl_file, 'r', encoding='utf-8') as f:
        #         async for line in f:
        #             try:
        #                 result = json.loads(line.strip())
        #                 if result['title'] in processed_ids:
        #                     results.append(result)
        #             except json.JSONDecodeError:
        #                 print(f"Warning: Skipping invalid JSON line in {self.config.jsonl_file}")

        return results

    async def score_research_proposal(self, data: List[Any]) -> List[Dict]:
        """
        Score abstracts using the configured model type.

        Args:
            data (List[Dict]): List of dictionaries containing 'title', 'abstract', and 'gt_score'.

        Returns:
            List[Dict]: List of results with 'title', 'score', 'evaluation', 'abstract', and 'gt_score'.
        """
        if isinstance(data[0], str):
            data = [{"title": "", "gt_score":0, "abstract": d} for d in data]

        if self.config.model_type == "vllm":
            return await self.score_with_vllm(data)
        elif self.config.model_type == "api":
            return await self.score_with_api(data)
        else:
            raise ValueError("Invalid model_type. Must be 'vllm' or 'api'.")

if __name__ == "__main__":
    async def test_reward_model():
        # Sample data for testing
        sample_data = [
            {
                "title": "A Novel Approach to Quantum Computing",
                "abstract": "This proposal introduces a new quantum algorithm that enhances computational efficiency by leveraging entangled states in a scalable architecture. The approach is validated through simulations showing a 20% improvement over existing methods.",
                "gt_score": 7.5
            },
            {
                "title": "AI-Driven Climate Modeling",
                "abstract": "We propose an AI-based framework for improving climate predictions using deep learning to integrate heterogeneous environmental data. Preliminary results demonstrate enhanced accuracy in long-term forecasts.",
                "gt_score": 8.0
            }
        ]

        # Test with vLLM model (commented out because it requires a local model and GPU)

        # try:
        #     vllm_model = RewardModel(
        #         model_type="vllm",
        #         model_name="/data/zhuotaodeng/yzj/alpha_research_model/qwen25_grm_iclr_boxed/checkpoint-180",
        #         jsonl_file="vllm_results.jsonl"
        #     )
        #     vllm_results = await vllm_model.score_research_proposal(sample_data)
        #     print("vLLM Results:")
        #     for result in vllm_results:
        #         print(f"Title: {result['title']}, Score: {result['score']}, Evaluation: {result['evaluation'][:50]}...")
        # except Exception as e:
        #     print(f"vLLM test failed: {e}")


        # Test with API model (requires valid API key and base URL)
        try:
            # Replace with your actual API key and base URL
            api_key = "sk-QooBWhqUg9xsdjHIxs1G0sVUIDB3h7QEIecPHK6WDNpNPlz2"
            base_url = "https://api.moonshot.cn/v1"

            config = RewardModelConfig(
                model_type="api",
                model_name="kimi-k2-0711-preview",
                api_key=api_key,
                base_url=base_url,
                jsonl_file="api_results.jsonl",
                max_retries=3,
                retry_delay=1
            ) 
            
            api_model = RewardModel(config)
            api_results = await api_model.score_research_proposal(sample_data)
            print("API Results:")
            for result in api_results:
                print(f"Title: {result['title']}, Score: {result['score']}, Evaluation: {result['evaluation'][:50]}...")
        except Exception as e:
            print(f"API test failed: {e}")

    # Run the async test
    asyncio.run(test_reward_model())
