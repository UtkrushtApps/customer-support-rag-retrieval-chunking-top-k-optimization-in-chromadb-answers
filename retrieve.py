import chromadb
from chromadb.config import Settings
from ingest_and_embed import SupportEmbedder

CHROMA_COLLECTION_NAME = 'support_chunks_v2'

class ChromaRAGRetriever:
    def __init__(self, collection_name=CHROMA_COLLECTION_NAME, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.coll = self.client.get_collection(collection_name)
        self.embedder = SupportEmbedder(model_name)

    def retrieve(self, query, top_k=5):
        query_emb = self.embedder.embed([query])[0]
        res = self.coll.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            include=["embeddings", "documents", "metadatas", "distances"]  # Get all info
        )
        results = []
        for i in range(len(res['ids'][0])):
            result = {
                'doc_id': res['metadatas'][0][i]['doc_id'],
                'category': res['metadatas'][0][i]['category'],
                'priority': res['metadatas'][0][i]['priority'],
                'date': res['metadatas'][0][i]['date'],
                'chunk_idx': res['metadatas'][0][i]['chunk_idx'],
                'document': res['documents'][0][i],
                'distance': res['distances'][0][i],
                'chunk_id': res['ids'][0][i]
            }
            results.append(result)
        return results

if __name__ == "__main__":
    retriever = ChromaRAGRetriever()
    sample_query = "How do I increase my subscription tier on the billing portal?"
    results = retriever.retrieve(sample_query, top_k=5)
    print("Top-5 Support Chunks for Query:")
    for i, r in enumerate(results):
        print(f"Rank {i+1}:")
        print(f" Doc ID: {r['doc_id']}, Category: {r['category']}, Priority: {r['priority']}, Date: {r['date']}")
        print(f" Chunk: {r['document'][:200]} ...")
        print(f" Distance: {r['distance']:.4f}")
        print()
