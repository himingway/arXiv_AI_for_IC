"""Shared LLM client factory.

Creates OpenAI-compatible or Anthropic client instances from environment config.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')


def create_llm_client(timeout: float = 120):
    """Create an LLM client based on LLM_PROVIDER env var.

    Returns:
        (provider, client, model, temperature, max_tokens)
    """
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    api_key = os.getenv('API_KEY', '')
    model = os.getenv('LLM_MODEL', 'gpt-4o')
    temperature = float(os.getenv('TEMPERATURE', '0.1'))

    if provider == 'openai':
        base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
    elif provider == 'anthropic':
        base_url = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
        client = anthropic.Anthropic(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'anthropic'.")

    return provider, client, model, temperature, api_key
