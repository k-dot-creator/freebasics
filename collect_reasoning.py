#!/usr/bin/env python3
from collector_base import ConceptCollector
import asyncio
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN environment variable not set")

DATASETS = [
    {"name": "gsm8k", "path": "openai/gsm8k"},
    {"name": "prm800k", "path": "openai/prm800k"},
    {"name": "commonsense_qa", "path": "tau/commonsense_qa"},
    {"name": "logical_fallacies", "path": "hails/logical-fallacies"},
]

async def main():
    collector = ConceptCollector("reasoning", DATASETS, HF_TOKEN, buffer_size_gb=3)
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())
