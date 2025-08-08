import json
from retrieve import ChromaRAGRetriever
from tqdm import tqdm

# Example input format for queries_with_answers.jsonl:
# {"query": "How to reset my password?", "relevant_doc_ids": ["doc1213", "doc0987"]}

QUERIES_WITH_ANSWERS = "queries_with_answers.jsonl"


def load_queries_with_answers(path):
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            item = json.loads(l.strip())
            queries.append(item)
    return queries

def recall_at_k(retriever, queries, k=5):
    """
    For each query, did at least one of the retrieved chunks correspond to a relevant doc_id?
    """
    hits = 0
    for q in tqdm(queries, desc="Recall@5 Evaluation"):
        results = retriever.retrieve(q['query'], top_k=k)
        retrieved_doc_ids = set([r['doc_id'] for r in results])
        if any(doc in retrieved_doc_ids for doc in q['relevant_doc_ids']):
            hits += 1
    return hits / len(queries)

if __name__ == "__main__":
    print("Evaluating Recall@5...")
    queries = load_queries_with_answers(QUERIES_WITH_ANSWERS)
    retriever = ChromaRAGRetriever()
    recall = recall_at_k(retriever, queries, k=5)
    print(f"Recall@5: {recall*100:.2f}% over {len(queries)} test queries.")
