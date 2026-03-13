#!/usr/bin/env python3
from collector_base import ConceptCollector
import asyncio
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN environment variable not set")

DATASETS = [
    {"name": "math_qa", "path": "math_qa"},
    {"name": "openwebmath", "path": "open-web-math/open-web-math"},
    {"name": "mathpile", "path": "AI-MO/MathPile"},
    {"name": "gsm8k", "path": "openai/gsm8k"},
    {"name": "math_dataset", "path": "deepmind/math_dataset"},
    {"name": "competition_math", "path": "competition_math"},
    {"name": "prm800k", "path": "openai/prm800k"},
]

async def main():
    collector = ConceptCollector("maths", DATASETS, HF_TOKEN, buffer_size_gb=3)
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())
