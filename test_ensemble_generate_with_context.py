#!/usr/bin/env python3
"""
Test script for LLMEnsemble.generate_with_context method
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict

# Add the parent directory to sys.path to import evolve_agent modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evolve_agent.config import load_config, RewardModelConfig
from evolve_agent.llm.ensemble import LLMEnsemble
from evolve_agent.config import LLMModelConfig


async def test_basic_functionality():
    """Test basic functionality of generate_with_context"""
    print("Testing basic functionality...")
    
    # Create mock model configurations
    model_configs = [
        LLMModelConfig(
            name="deepseek-chat",
            api_key="sk-2c3f1f58031b4b86afdb6a8192ea02e2",
            api_base="https://api.deepseek.com",
            weight=1.0,
            temperature=0.7,
            max_tokens=1000,
            timeout=60,
            retries=3,
            retry_delay=1
        )
    ]
    model_configs = load_config("configs/default_config.yaml")
    # Create ensemble
    ensemble = LLMEnsemble(model_configs.llm.models)
    
    # Mock the OpenAI API calls
    with patch('evolve_agent.llm.openai.AsyncOpenAI') as mock_openai:
        simple_idea = "挂谷猜想"
        proposal_prompt = f"""
Please write a paragraph of proposal for the idea focusing on the following points: 

1.clarity 
2.novelty 
3.exact method technically
4.technical rigor 
5.potential impact

挂谷猜想
"""
        # Test parameters
        system_message = "You are a helpful assistant."
        messages = [
            {"role": "user", "content": proposal_prompt},
            # {"role": "assistant", "content": "I'm doing well, thank you!"},
            # {"role": "user", "content": "Can you help me with a task?"}
        ]

 

        # try:
        #     response = await ensemble.generate_with_context(
        #         system_message="You are an expert research proposal writer with extensive experience in computer science and algorithm development.",
        #         messages=[{"role": "user", "content": proposal_prompt}]
        #     )
        #     return response.strip()
        # except Exception as e:
        #     print(f"Error generating proposal: {e}")
        #     return f"Research Proposal: {simple_idea}\n\nThis research aims to explore and develop solutions for {simple_idea}."

        # Call generate_with_context
        result = await ensemble.generate_with_context(system_message, messages)
        
        # Verify result
        print(result)
        assert 1==2
        
        # Check that system message was added
        expected_messages = [{"role": "system", "content": system_message}] + messages
        assert call_args["messages"] == expected_messages
        print("✓ System message properly added")
        
        # Check that model parameters were passed
        assert "model" in call_args
        assert call_args["model"] in ["gpt-4", "gpt-3.5-turbo"]
        print("✓ Model parameters properly passed")


async def test_model_selection():
    """Test that ensemble selects models based on weights"""
    print("\nTesting model selection...")
    
    # Create model configurations with different weights
    model_configs = [
        LLMModelConfig(
            name="model-a",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            weight=0.8,
            temperature=0.7,
            max_tokens=1000,
            timeout=60,
            retries=3,
            retry_delay=1
        ),
        LLMModelConfig(
            name="model-b",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            weight=0.2,
            temperature=0.7,
            max_tokens=1000,
            timeout=60,
            retries=3,
            retry_delay=1
        )
    ]
    
    ensemble = LLMEnsemble(model_configs)
    
    # Track which models are selected
    selected_models = []
    
    with patch('evolve_agent.llm.openai.AsyncOpenAI') as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Make multiple calls to test distribution
        for i in range(20):
            await ensemble.generate_with_context(
                "System message",
                [{"role": "user", "content": f"Test message {i}"}]
            )
            
            # Get the model name from the last call
            call_args = mock_client.chat.completions.create.call_args[1]
            selected_models.append(call_args["model"])
    
    # Check that model-a was selected more often (it has higher weight)
    model_a_count = selected_models.count("model-a")
    model_b_count = selected_models.count("model-b")
    
    print(f"Model A selected: {model_a_count} times")
    print(f"Model B selected: {model_b_count} times")
    
    # With weight 0.8 vs 0.2, model-a should be selected more often
    assert model_a_count > model_b_count
    print("✓ Model selection based on weights works correctly")


async def test_error_handling():
    """Test error handling and retries"""
    print("\nTesting error handling...")
    
    model_configs = [
        LLMModelConfig(
            name="test-model",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            weight=1.0,
            temperature=0.7,
            max_tokens=1000,
            timeout=1,  # Short timeout for testing
            retries=2,
            retry_delay=0.1
        )
    ]
    
    ensemble = LLMEnsemble(model_configs)
    
    with patch('evolve_agent.llm.openai.AsyncOpenAI') as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Mock API to raise timeout error
        mock_client.chat.completions.create.side_effect = asyncio.TimeoutError()
        
        # Test that timeout error is properly raised after retries
        try:
            await ensemble.generate_with_context(
                "System message",
                [{"role": "user", "content": "Test"}]
            )
            assert False, "Should have raised TimeoutError"
        except asyncio.TimeoutError:
            print("✓ Timeout error properly raised")
        
        # Verify that retries were attempted
        # Should be called 3 times (1 initial + 2 retries)
        assert mock_client.chat.completions.create.call_count == 3
        print("✓ Retry mechanism works correctly")


async def test_realistic_scenario():
    """Test with realistic parameters and conversation"""
    print("\nTesting realistic scenario...")
    
    model_configs = [
        LLMModelConfig(
            name="gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            weight=0.6,
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            timeout=60,
            retries=3,
            retry_delay=2
        ),
        LLMModelConfig(
            name="claude-3-sonnet",
            api_key="test-key",
            api_base="https://api.anthropic.com/v1",
            weight=0.4,
            temperature=0.8,
            top_p=0.95,
            max_tokens=4096,
            timeout=90,
            retries=2,
            retry_delay=3
        )
    ]
    
    ensemble = LLMEnsemble(model_configs)
    
    with patch('evolve_agent.llm.openai.AsyncOpenAI') as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Mock realistic API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Based on your conversation history, I can see you're working on a machine learning project. 
        Here are some suggestions for improving your model performance:
        
        1. Feature engineering: Consider creating polynomial features
        2. Hyperparameter tuning: Use grid search or random search
        3. Cross-validation: Implement k-fold cross-validation
        4. Regularization: Try L1 or L2 regularization
        
        Would you like me to elaborate on any of these techniques?
        """
        mock_client.chat.completions.create.return_value = mock_response
        
        # Realistic conversation
        system_message = "You are an expert machine learning engineer helping with model optimization."
        messages = [
            {"role": "user", "content": "I'm working on a regression model but getting poor performance."},
            {"role": "assistant", "content": "I'd be happy to help! Can you tell me more about your dataset and current approach?"},
            {"role": "user", "content": "I have 10,000 samples with 20 features. Using linear regression with R² of 0.65."},
            {"role": "assistant", "content": "That's a decent baseline. What's your target variable and feature types?"},
            {"role": "user", "content": "Predicting house prices. Mix of numerical (area, bedrooms) and categorical (neighborhood, style)."},
            {"role": "user", "content": "How can I improve this model?"}
        ]
        
        # Call with custom parameters
        result = await ensemble.generate_with_context(
            system_message, 
            messages,
            temperature=0.6,
            max_tokens=1500
        )
        
        # Verify result
        assert result is not None
        assert len(result) > 0
        print("✓ Realistic scenario test passed")
        
        # Verify API call parameters
        call_args = mock_client.chat.completions.create.call_args[1]
        
        # Check that custom parameters were used
        assert call_args["temperature"] == 0.6
        assert call_args["max_tokens"] == 1500
        print("✓ Custom parameters properly passed")
        
        # Check message formatting
        expected_messages = [{"role": "system", "content": system_message}] + messages
        assert call_args["messages"] == expected_messages
        print("✓ Message formatting correct")


async def main():
    """Run all tests"""
    print("Testing LLMEnsemble.generate_with_context")
    print("=" * 50)
    
    try:
        await test_basic_functionality()
        await test_model_selection()
        await test_error_handling()
        await test_realistic_scenario()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())