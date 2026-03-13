#!/usr/bin/env python3
from collector_base import ConceptCollector
import asyncio
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN environment variable not set")

DATASETS = [
    {"name": "arxiv_dataset", "path": "arxiv_dataset"},
    {"name": "pubmed", "path": "pubmed"},
    {"name": "s2orc", "path": "allenai/s2orc"},
    {"name": "cord19", "path": "allenai/cord19"},
]

async def main():
    collector = ConceptCollector("scientific", DATASETS, HF_TOKEN, buffer_size_gb=3)
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())
