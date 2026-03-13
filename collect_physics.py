#!/usr/bin/env python3
from collector_base import ConceptCollector
import asyncio
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN environment variable not set")

DATASETS = [
    {"name": "arxiv_physics", "path": "arxiv_dataset", "filter": "physics"},
    {"name": "quantum_computing_qa", "path": "qedma/quantum_computing_qa"},
    {"name": "cern_open_data", "path": "cern/opendata"},
    {"name": "hep_papers", "path": "arxiv-hep-th"},
    {"name": "quantum_mechanics_textbooks", "path": "Omartificial-Intelligence/Quantum-Mechanics-Textbooks"},
    {"name": "qiskit_docs", "path": "alexjercan/qiskit-documentation"},
    {"name": "science_qa", "path": "derek-thomas/ScienceQA"},
]

async def main():
    collector = ConceptCollector("physics", DATASETS, HF_TOKEN, buffer_size_gb=3)
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())
