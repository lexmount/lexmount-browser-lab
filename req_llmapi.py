# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random

import openai

"""
export OPENAI_API_KEY=sk-fgrNlns4poTwmzABx8IP-A
export OPENAI_BASE_URL=https://litellm.local.lexmount.net/v1
export OPENAI_MODEL=o4-mini
python req_llmapi.py

export OPENAI_API_KEY=sk-abc123
export OPENAI_BASE_URL=http://10.2.131.41:18088/v1
export OPENAI_MODEL=qwen3_8B
python req_llmapi.py
"""

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


OPENAI_API_KEY = require_env("OPENAI_API_KEY")
OPENAI_BASE_URL = require_env("OPENAI_BASE_URL")
OPENAI_MODEL = require_env("OPENAI_MODEL")

print(OPENAI_API_KEY)
print(OPENAI_BASE_URL)
print(OPENAI_MODEL)
print("--------------------------------")


class OpenAIAPI:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, *, user_prompt: str, system_prompt: str | None = None) -> str:
        if not system_prompt:
            system_prompt = "You are a helpful assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        _seed = random.randint(0, 1000000)
        completion = self.client.chat.completions.create(messages=messages, model=self.model)
        return str(completion.choices[0].message.content)


openai_api = OpenAIAPI(OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL)

if __name__ == "__main__":
    print(
        openai_api.generate(
            user_prompt="你谁啊？你知道世界三大暴君吗",
            system_prompt="You are a helpful assistant.",
        )
    )
