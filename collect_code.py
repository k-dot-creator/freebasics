#!/usr/bin/env python3
from collector_base import ConceptCollector
import asyncio
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN environment variable not set")

DATASETS = [
    {"name": "the_stack", "path": "bigcode/the-stack-dedup"},
    {"name": "code_search_net", "path": "code_search_net"},
    {"name": "codeparrot_github", "path": "codeparrot/github-code"},
    {"name": "apps", "path": "codeparrot/apps"},
    {"name": "code_contests", "path": "deepmind/code_contests"},
]

async def main():
    collector = ConceptCollector("code", DATASETS, HF_TOKEN, buffer_size_gb=3)
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())
