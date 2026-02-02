import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Optional: OpenAI or Upstage API for embeddings
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

class RAGSystem:
    def __init__(self, use_api=False, model_name='BAAI/bge-m3', index_path='models/faiss_index.bin'):
        self.use_api = use_api or os.getenv("EMBEDDING_API_KEY") is not None
        self.index_path = index_path
        self.documents = []
        self.index = None
        
        if not self.use_api:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading local embedding model: {model_name} on {device}")
            self.model = SentenceTransformer(model_name, device=device)
        else:
            api_key = os.getenv("EMBEDDING_API_KEY")
            base_url = os.getenv("EMBEDDING_API_BASE", "https://api.openai.com/v1")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.api_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
            print(f"Using API for embeddings: {self.api_model} ({base_url})")

    def _get_embeddings(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        if not self.use_api:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
        else:
            # Batch processing for API to avoid rate limits/timeouts
            batch_size = 100
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(input=batch, model=self.api_model)
                all_embeddings.extend([data.embedding for data in response.data])
            return np.array(all_embeddings).astype('float32')

    def build_index(self, corpus_texts: List[str]):
        import torch
        print(f"Encoding {len(corpus_texts)} documents...")
        embeddings = self._get_embeddings(corpus_texts, show_progress=True)
        
        self.documents = corpus_texts
        self.doc_embeddings = torch.from_numpy(embeddings).float()
        
        if torch.cuda.is_available():
            self.doc_embeddings = self.doc_embeddings.to("cuda")
            print("Document embeddings moved to GPU (PyTorch).")
        
        # Save embeddings locally
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        torch.save(self.doc_embeddings.cpu(), f"{self.index_path}.pt")
        with open(f"{self.index_path}.docs.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False)
        print("Index (PyTorch) build complete and saved.")

    def load_index(self):
        import torch
        if os.path.exists(f"{self.index_path}.pt"):
            self.doc_embeddings = torch.load(f"{self.index_path}.pt")
            if torch.cuda.is_available():
                self.doc_embeddings = self.doc_embeddings.to("cuda")
                print("Index (PyTorch) loaded and moved to GPU.")
            
            with open(f"{self.index_path}.docs.json", 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print("Index loaded.")
        else:
            print(f"Index file not found.")

    def retrieve(self, query: str, top_k=10):
        results = self.bulk_retrieve([query], top_k=top_k)
        return results[0]

    def bulk_retrieve(self, queries: List[str], top_k=10):
        import torch
        if not hasattr(self, 'doc_embeddings'):
            self.load_index()
            
        # 1. Get query embeddings
        q_embeddings = self._get_embeddings(queries, show_progress=False)
        q_embeddings = torch.from_numpy(q_embeddings).float().to(self.doc_embeddings.device)
        
        # 2. Compute similarity (Vector MatMul)
        # Assuming bge-m3 uses cosine sim (effectively Dot Product because normalized)
        # Normalize for cosine similarity
        q_norm = torch.nn.functional.normalize(q_embeddings, p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(self.doc_embeddings, p=2, dim=1)
        
        # Matrix multiplication: [batch, dim] @ [dim, num_docs] -> [batch, num_docs]
        scores = torch.mm(q_norm, doc_norm.t())
        
        # 3. Get top-k
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(self.documents)), dim=1)
        
        batch_results = []
        for i in range(len(queries)):
            query_res = []
            for j in range(len(top_indices[i])):
                idx = int(top_indices[i][j])
                query_res.append({
                    "text": self.documents[idx],
                    "score": float(top_scores[i][j])
                })
            batch_results.append(query_res)
        return batch_results

if __name__ == "__main__":
    # Test with local or API (depending on .env)
    rag = RAGSystem()
    sample_docs = ["금융 상품은 위험이 있습니다.", "법률은 평등합니다."]
    rag.build_index(sample_docs)
    res = rag.retrieve("위험한 거 있어?")
    for r in res:
        print(f"Score: {r['score']:.4f} | Text: {r['text']}")
