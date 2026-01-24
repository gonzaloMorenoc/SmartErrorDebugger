import os
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

class LogLoader:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )

    def _process_json_file(self, file_path):
        """Processes a single JSON file and converts it into a Document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract relevant fields for context
            # We try to build a meaningful string from common error report fields
            content_parts = []
            
            # Priority fields
            if "error_message" in data: content_parts.append(f"Error: {data['error_message']}")
            if "stack_trace" in data: content_parts.append(f"Stack Trace: {data['stack_trace']}")
            if "previous_fix" in data: content_parts.append(f"Solution: {data['previous_fix']}")
            
            # If it's a generic JSON, flatten it
            if not content_parts:
                content_parts.append(json.dumps(data, indent=2))
            
            content = "\n".join(content_parts)
            return Document(page_content=content, metadata={"source": file_path, "type": "json"})
        except Exception as e:
            print(f"Error processing JSON {file_path}: {e}")
            return None

    def load(self):
        """Loads logs and error reports from the data directory."""
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return []

        all_docs = []

        # 1. Load .log files (Standard text loading)
        log_loader = DirectoryLoader(self.data_path, glob="./*.log", loader_cls=TextLoader)
        all_docs.extend(log_loader.load())

        # 2. Load and process .json files (Custom logic)
        for filename in os.listdir(self.data_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_path, filename)
                doc = self._process_json_file(file_path)
                if doc:
                    all_docs.append(doc)
        
        if not all_docs:
            return []

        return self.text_splitter.split_documents(all_docs)
