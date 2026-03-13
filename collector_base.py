# collector_base.py
import asyncio
import aiohttp
from huggingface_hub import HfApi, HfFileSystem, login, create_repo, repo_exists
from datasets import load_dataset
import json
import os
import time
from datetime import datetime
import uuid
import gzip
import sys

class ConceptCollector:
    def __init__(self, concept_name, dataset_list, hf_token, buffer_size_gb=3):
        self.concept = concept_name
        self.datasets = dataset_list
        self.token = hf_token
        self.repo_id = f"RobbieJr/biro-ai-{concept_name}-dataset"
        self.repo_type = "dataset"
        self.checkpoint_file = f"checkpoint_{concept_name}.json"
        self.buffer_size_bytes = buffer_size_gb * 1024**3
        self.max_concurrent = 5
        self.compression_level = 6
        self.timeout_hours = 5.5
        self.batch_samples = 100000  # fallback

        self.api = HfApi()
        self.fs = HfFileSystem(token=self.token)
        self.session = None
        self.buffer = []
        self.buffer_size = 0
        self.processed_sources = {}
        self.start_time = time.time()
        self.stats = {"total_samples": 0, "total_bytes": 0, "errors": []}

        # Ensure repo exists
        login(token=self.token, add_to_git_credential=True)
        if not repo_exists(repo_id=self.repo_id, repo_type=self.repo_type, token=self.token):
            print(f"📦 Creating repository: {self.repo_id}")
            create_repo(repo_id=self.repo_id, repo_type=self.repo_type, token=self.token, private=False, exist_ok=True)
            time.sleep(2)
        print(f"✅ Repository ready: {self.repo_id}")

        self._load_checkpoint()

    def _load_checkpoint(self):
        try:
            cp_path = f"{self.repo_id}/{self.checkpoint_file}"
            if self.fs.exists(cp_path):
                with self.fs.open(cp_path, "rb") as f:
                    data = json.loads(f.read().decode())
                    self.processed_sources = data.get("processed_sources", {})
                    self.stats["total_samples"] = data.get("total_samples", 0)
                    self.stats["total_bytes"] = data.get("total_bytes", 0)
                    print(f"✅ Loaded checkpoint: {len(self.processed_sources)} sources processed")
            else:
                print("ℹ️  No checkpoint found, starting fresh")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")

    def _save_checkpoint(self):
        cp = {
            "processed_sources": self.processed_sources,
            "total_samples": self.stats["total_samples"],
            "total_bytes": self.stats["total_bytes"],
            "last_run": datetime.now().isoformat()
        }
        data = json.dumps(cp, indent=2).encode()
        try:
            self.api.upload_file(
                path_or_fileobj=data,
                path_in_repo=self.checkpoint_file,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.token
            )
            print("✅ Checkpoint saved")
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")

    def should_stop(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_hours * 3600:
            print(f"\n⏰ Time limit reached ({elapsed/3600:.1f} hours). Stopping.")
            return True
        return False

    async def init_session(self):
        timeout = aiohttp.ClientTimeout(total=3600)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, force_close=True)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"Authorization": f"Bearer {self.token}"}
        )

    async def close_session(self):
        if self.session:
            await self.session.close()

    async def upload_with_retry(self, compressed_data, filename):
        for attempt in range(1, 6):
            try:
                self.api.upload_file(
                    path_or_fileobj=compressed_data,
                    path_in_repo=filename,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    token=self.token
                )
                return True
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait = 5 * (2 ** (attempt - 1))
                    print(f"    ⏳ Rate limited. Retry {attempt}/5 after {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise e
        return False

    async def flush_buffer(self):
        if not self.buffer:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        filename = f"{self.concept}/batch_{batch_id}.jsonl.gz"

        jsonl_data = "\n".join(json.dumps(s) for s in self.buffer) + "\n"
        compressed = gzip.compress(jsonl_data.encode("utf-8"), compresslevel=self.compression_level)

        print(f"    📦 Flushing {len(self.buffer)} samples ({self.buffer_size/1024/1024:.2f} MB raw)")
        success = await self.upload_with_retry(compressed, filename)
        if success:
            self.stats["total_bytes"] += len(compressed)
            print(f"    📤 Uploaded to {filename} ({len(compressed)/1024/1024:.2f} MB compressed)")
        else:
            self.stats["errors"].append(f"Upload failed: {filename}")
            print(f"    ❌ Upload failed after retries")

        self.buffer = []
        self.buffer_size = 0

    async def add_to_buffer(self, sample):
        sample_json = json.dumps(sample)
        sample_size = len(sample_json.encode('utf-8'))

        self.buffer.append(sample)
        self.buffer_size += sample_size

        if self.buffer_size >= self.buffer_size_bytes:
            print(f"    📊 Buffer reached {self.buffer_size/1024/1024/1024:.2f}GB – flushing")
            await self.flush_buffer()
        elif len(self.buffer) >= self.batch_samples:
            print(f"    📊 Buffer reached {len(self.buffer)} samples – flushing")
            await self.flush_buffer()

        self.stats["total_samples"] += 1

    async def stream_dataset(self, name, path, filter_key=None):
        if name in self.processed_sources:
            print(f"  ⏩ Skipping {name} (already processed)")
            return 0

        print(f"\n  📦 Processing: {name} ({path})")
        samples_collected = 0

        try:
            ds = load_dataset(path, split="train", streaming=True, trust_remote_code=True)
            for item in ds:
                if self.should_stop():
                    break

                if filter_key and filter_key not in str(item).lower():
                    continue

                text = self._extract_text(item)
                if text and len(text) > 10:
                    sample = {
                        "text": text[:100000],
                        "metadata": {
                            "source": name,
                            "original_path": path,
                            "concept": self.concept,
                            "collected_at": datetime.now().isoformat(),
                            "sample_id": str(uuid.uuid4())
                        }
                    }
                    await self.add_to_buffer(sample)
                    samples_collected += 1

                    if samples_collected % 5000 == 0:
                        buffer_gb = self.buffer_size / (1024**3)
                        print(f"     Progress: {samples_collected:,} | Buffer: {buffer_gb:.2f}GB")

            await self.flush_buffer()
            self.processed_sources[name] = samples_collected
            print(f"    ✅ Completed: {samples_collected:,} samples")
        except Exception as e:
            self.stats["errors"].append(f"{name}: {str(e)}")
            print(f"    ❌ Error: {e}")
        return samples_collected

    def _extract_text(self, item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for field in ['text', 'content', 'article', 'abstract', 'body', 'description', 'code']:
                if field in item and isinstance(item[field], str):
                    return item[field]
            if 'question' in item and 'answer' in item:
                return f"Question: {item['question']}\n\nAnswer: {item['answer']}"
            if 'instruction' in item and 'output' in item:
                return f"Instruction: {item['instruction']}\n\nOutput: {item['output']}"
            try:
                return json.dumps(item)
            except:
                return str(item)
        return str(item)

    async def run(self):
        await self.init_session()
        for ds in self.datasets:
            if self.should_stop():
                break
            name = ds["name"]
            path = ds["path"]
            filter_key = ds.get("filter")
            await self.stream_dataset(name, path, filter_key)
        await self.flush_buffer()
        await self.close_session()
        self._save_checkpoint()

        elapsed = time.time() - self.start_time
        print("\n" + "="*80)
        print(f"{self.concept.upper()} SESSION SUMMARY")
        print("="*80)
        print(f"⏱️  Runtime: {elapsed/3600:.2f} hours")
        print(f"📊 New samples this run: {self.stats['total_samples'] - self.processed_sources.get('_last_total', 0):,}")
        print(f"📦 Total samples overall: {self.stats['total_samples']:,}")
        print(f"💾 Total bytes: {self.stats['total_bytes'] / (1024**4):.2f} TB")
        if self.stats['errors']:
            print(f"⚠️  Errors: {len(self.stats['errors'])}")
