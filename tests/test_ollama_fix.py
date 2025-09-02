#!/usr/bin/env python3
"""
Test script to verify the Ollama transformation fix for handling thinking and tool_calls fields.
"""

import json
import sys
import os

# Add the litellm directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'litellm'))

import tiktoken
from litellm.llms.ollama.completion.transformation import OllamaConfig
from litellm.types.utils import ModelResponse, Choices, Message
from litellm.types.llms.openai import ChatCompletionUsageBlock

def test_transform_response():
    """Test the transform_response method with the problematic response."""
    
    # Create the config
    config = OllamaConfig()
    
    # Create a mock response object
    class MockResponse:
        def json(self):
            return {
                'model': 'gpt-oss:20b', 
                'created_at': '2025-09-01T23:15:25.443780589Z', 
                'response': '', 
                'thinking': "We need to run `ls -la` in the system. The environment says we have access to bash tools. We should output the result of the command. Let's run it.", 
                'done': True, 
                'done_reason': 'stop', 
                'load_duration': 147465232, 
                'prompt_eval_count': 121, 
                'prompt_eval_duration': 855247334, 
                'eval_count': 65, 
                'eval_duration': 5377148835, 
                'tool_calls': [{'function': {'name': 'container.exec', 'arguments': {'cmd': ['bash', '-lc', 'ls -la']}}}]
            }
    
    # Create a mock model response
    model_response = ModelResponse(
        id='test-id',
        created=0,
        model='ollama/gpt-oss:20b',
        object='chat.completion',
        choices=[Choices(
            finish_reason='stop',
            index=0,
            message=Message(content='', role='assistant')
        )],
        usage=ChatCompletionUsageBlock(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
    )
    
    # Mock other required parameters
    class MockLoggingObj:
        pass
    
    logging_obj = MockLoggingObj()
    request_data = {'prompt': 'test prompt'}
    messages = []
    optional_params = {}
    litellm_params = {}
    encoding = tiktoken.get_encoding("cl100k_base")  # Use proper tiktoken encoding
    
    # Call the transform_response method
    result = config.transform_response(
        model='gpt-oss:20b',
        raw_response=MockResponse(),
        model_response=model_response,
        logging_obj=logging_obj,
        request_data=request_data,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        encoding=encoding
    )
    
    # Print the results
    print("=== Test Results ===")
    print(f"Model: {result.model}")
    print(f"Finish Reason: {result.choices[0].finish_reason}")
    print(f"Content: {result.choices[0].message.content}")
    print(f"Tool Calls: {result.choices[0].message.tool_calls}")
    print(f"Usage - Prompt Tokens: {result.usage.prompt_tokens}")
    print(f"Usage - Completion Tokens: {result.usage.completion_tokens}")
    print(f"Usage - Total Tokens: {result.usage.total_tokens}")
    
    # Verify the results
    assert result.choices[0].finish_reason == "tool_calls", f"Expected 'tool_calls', got '{result.choices[0].finish_reason}'"
    assert result.choices[0].message.tool_calls is not None, "Expected tool_calls to be present"
    assert len(result.choices[0].message.tool_calls) == 1, f"Expected 1 tool call, got {len(result.choices[0].message.tool_calls)}"
    assert result.choices[0].message.tool_calls[0]["function"]["name"] == "container.exec", "Expected function name 'container.exec'"
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_transform_response()
