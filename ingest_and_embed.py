import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch
import json
from tqdm import tqdm
from chunker import chunk_text

class SupportEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def embed(self, texts):
        # Batch embedding
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
        return embeddings.cpu().numpy()
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element: output features
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Modify path as needed.
SUPPORT_DOCS_PATH = 'support_articles.jsonl'
CHROMA_COLLECTION_NAME = 'support_chunks_v2'

# 1. Prepare ChromaDB Client and Collection
def get_chroma_collection(collection_name):
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    # Drop old collection if exists
    if collection_name in [c.name for c in client.list_collections()]:
        client.get_collection(collection_name).delete()
        client.delete_collection(collection_name)
    # Create new collection
    return client.create_collection(collection_name)

def load_support_docs(path):
    """
    Assume each line in support_articles.jsonl is a JSON object with fields:
    {
        "id": "unique_id",
        "category": "Billing",
        "priority": "High",
        "date": "2023-10-11",
        "text": "Support document full text..."
    }
    """
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            docs.append(obj)
    return docs

def chunk_and_embed_articles(docs, collection, embedder, chunk_size=200, overlap=50, batch_size=32):
    doc_ids = []
    chunk_texts = []
    chunk_metas = []
    chunk_id_counter = 0
    batch_embeds = []
    batch_ids = []
    batch_metas = []
    batch_texts = []
    for doc in tqdm(docs, desc="Chunking and embedding"):
        chunks = chunk_text(doc['text'], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk{i}"
            meta = {
                'doc_id': doc['id'],
                'category': doc.get('category'),
                'priority': doc.get('priority'),
                'date': doc.get('date'),
                'chunk_idx': i
            }
            batch_ids.append(chunk_id)
            batch_texts.append(chunk)
            batch_metas.append(meta)
            if len(batch_texts) == batch_size:
                embeds = embedder.embed(batch_texts)
                collection.add(
                    embeddings=embeds.tolist(),
                    documents=batch_texts,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                batch_ids, batch_metas, batch_texts = [], [], []
    # Last batch
    if batch_texts:
        embeds = embedder.embed(batch_texts)
        collection.add(
            embeddings=embeds.tolist(),
            documents=batch_texts,
            metadatas=batch_metas,
            ids=batch_ids
        )

if __name__ == "__main__":
    docs = load_support_docs(SUPPORT_DOCS_PATH)
    collection = get_chroma_collection(CHROMA_COLLECTION_NAME)
    embedder = SupportEmbedder()
    chunk_and_embed_articles(docs, collection, embedder)
    print("Finished ingestion and embedding of all support articles.")
