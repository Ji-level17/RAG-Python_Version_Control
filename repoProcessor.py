import os
import tqdm
import json
import hashlib

class RepoProcessor:
    def __init__(self, rag_instance):
        self.rag = rag_instance

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.processed_files, f)

    def _get_file_hash(self, file_path):
        """计算文件哈希，用于检测内容是否变更"""
        # compute file hash for change detection
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def process_repository(self, repo_path, repo_name):
        print(f"🚀 正在攻占仓库: {repo_name}...")
        
        py_files = []
        # 排除掉不相关的文件夹，只盯着核心源码
        # exclude unrelated folders, focus on core source code only
        excluded_dirs = {'.git', '__pycache__', 'tests', 'docs', 'benchmarks', 'site'}
        
        for root, dirs, files in os.walk(repo_path):
            # 跳过被排除的目录
            # skip excluded directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    py_files.append(os.path.join(root, file))

        print(f"统计：发现 {len(py_files)} 个有效 Python 文件。准备让 Qwen 开始加班...")

        for file_path in tqdm.tqdm(py_files, desc="Feeding code to Qwen"):
            try:
                # 1. 提取函数
                # 1. Extract functions
                functions = self.rag.extract_functions(file_path)
                rel_path = os.path.relpath(file_path, repo_path)
                
                for i, func_code in enumerate(functions):
                    # 2. 调用 python 3.12+ QLoRA 
                    # 2.use python 3.12+ QLoRA to enhance metadata
                    meta = self.rag.get_qwen_metadata(func_code)
                    
                    # 3. 增强元数据：存入路径，方便以后找回
                    # 3. Enhance metadata: store path for easy retrieval later
                    doc_id = f"{repo_name}_{rel_path.replace(os.sep, '_')}_f{i}"
                    summary = f"[{rel_path}] {meta.get('summary_zh', '')}"
                    
                    self.rag.ingest_data(
                        doc_id=doc_id,
                        code_content=func_code,
                        min_version=meta.get("min_version", "3.8"),
                        summary=summary
                    )
            except Exception as e:
                # 实战中总会有文件编码或其他奇怪问题，跳过它，不能让程序停下来
                # In practice, there will always be encoding issues or other weird problems with some files. Skip them, don't let the program stop.
                print(f"⚠️ 跳过文件 {file_path}: {e}") #skipping file due to error
                continue

        print(f"🏁 仓库 {repo_name} 摄取完成！") # print if successfully get from the git