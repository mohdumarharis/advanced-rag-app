"""Azure OpenAI LLM client factory."""
from functools import lru_cache

from langchain_openai import AzureChatOpenAI

import config


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.3) -> AzureChatOpenAI:
    """
    Return a cached AzureChatOpenAI client, built on first use.

    Deliberately not constructed at import time: doing so ran before
    config.validate() could report missing credentials, so a misconfigured
    .env surfaced as an opaque client error instead of a named-variable one.
    """
    config.validate()
    return AzureChatOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_KEY,
        api_version=config.AZURE_VERSION,
        azure_deployment=config.AZURE_DEPLOYMENT,
        temperature=temperature,
        max_tokens=100,
    )
