import os
import tqdm
import json
import hashlib

SAVE_STATE_INTERVAL = 10  # Persist state to disk every N successfully processed files


class RepoProcessor:
    def __init__(self, rag_instance, state_file="processed_state.json"):
        self.rag = rag_instance
        self.state_file = state_file
        self.processed_files = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.processed_files, f)

    def _get_file_hash(self, file_path):
        """Compute MD5 hash of a file for change detection."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def process_repository(self, repo_path, repo_name):
        """Scan a repository, ingest new or modified Python files, and rebuild the BM25 index."""
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"Repository path not found: {repo_path}")

        if self.processed_files and not self.rag.local_corpus:
            print("Warning: prior state detected but local corpus not loaded. Call rag.load_local_corpus() before processing.")

        print(f"Processing repository: {repo_name}...")

        py_files = []
        excluded_dirs = {'.git', '__pycache__', 'tests', 'docs', 'benchmarks', 'site'}

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for file in files:
                if file.endswith('.py') and not file.startswith('test_') and not file.endswith('_test.py'):
                    py_files.append(os.path.join(root, file))

        print(f"Found {len(py_files)} valid Python files.")

        skipped = 0
        processed = 0

        for file_path in tqdm.tqdm(py_files, desc="Ingesting"):
            rel_path = os.path.relpath(file_path, repo_path)
            file_hash = self._get_file_hash(file_path)

            if self.processed_files.get(rel_path) == file_hash:
                skipped += 1
                continue

            try:
                # Determine tier from relative path for stratified retrieval
                norm_path = rel_path.replace("\\", "/")
                if norm_path.startswith("docs_src/"):
                    tier = "docs_src"
                elif norm_path.startswith("fastapi/"):
                    tier = "fastapi"
                else:
                    tier = "scripts"

                # Skip scripts/ entirely — build tooling has no retrieval value
                # and acts as noise that crowds out tutorial results
                if tier == "scripts":
                    skipped += 1
                    continue

                functions = self.rag.extract_functions(file_path)

                # Normalise path separators so doc_ids are consistent across OS
                safe_path = rel_path.replace("\\", "/").replace("/", "_")

                for i, func_code in enumerate(functions):
                    meta = self.rag.get_qwen_metadata(func_code)
                    doc_id = f"{repo_name}_{safe_path}_f{i}"
                    summary_text = meta.get("summary") or meta.get("summary_zh", "")
                    summary = f"[{rel_path}] {summary_text}"

                    self.rag.ingest_data(
                        doc_id=doc_id,
                        code_content=func_code,
                        min_version=meta.get("min_version", "3.8"),
                        summary=summary,
                        tier=tier,
                    )

                self.processed_files[rel_path] = file_hash
                processed += 1

                if processed % SAVE_STATE_INTERVAL == 0:
                    self._save_state()

            except Exception as e:
                print(f"Warning: skipping {file_path}: {e}")
                continue

        # Flush any remaining buffered vectors and persist final state
        self.rag.flush_upsert_buffer()
        self._save_state()

        print(f"Ingestion complete. Processed {processed} files, skipped {skipped} unchanged.")

        if processed > 0:
            self.rag.rebuild_bm25()
        else:
            print("No new files processed, BM25 index unchanged.")
