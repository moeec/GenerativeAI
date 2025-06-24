#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 20:19:31 2025

@author: mwesim3
"""

# In your shell, first export the key once:
# export OPENAI_API_KEY="<key>"
# export OPENAI_API_BASE="https://openai.vocareum.com/v1"

import os
from openai import OpenAI

# 1. Load credentials
api_key  = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")  # or whatever env-var you’re using

if not api_key or not api_base:
    raise RuntimeError("Set OPENAI_API_KEY and OPENAI_API_BASE in your environment")

# 2. Instantiate the client with base_url, not api_base
client = OpenAI(
    api_key=api_key,
    base_url=api_base
)

# 3. Call the chat endpoint
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Write a one-sentence story about Mwesi's Superpowers."}]
)

# 4. Print the assistant’s reply
print(response.choices[0].message.content.strip())
