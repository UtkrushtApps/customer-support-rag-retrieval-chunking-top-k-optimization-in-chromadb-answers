# Solution Steps

1. Install required Python packages: chromadb, transformers, torch, tiktoken, tqdm, nltk. Ensure you have the actual support_articles.jsonl file and, for evaluation, queries_with_answers.jsonl.

2. Implement a chunking utility (chunker.py) that splits text into 200-token chunks with 50-token overlap, using tiktoken encoding for accurate tokenization.

3. Write the ingestion & embedding pipeline (ingest_and_embed.py): Read input articles, use the chunking utility to split them, attach all required metadata (category, priority, date, chunk_idx, doc_id), generate batch sentence embeddings (recommended: sentence-transformers/all-MiniLM-L6-v2), and add the chunks with metadata to a freshly created ChromaDB collection.

4. Add code in ingest_and_embed.py to drop any previous collection with same name to ensure a clean, deduplicated setup.

5. Implement a retrieval module (retrieve.py): For a given query, generate its embedding, perform a top-5 semantic search in ChromaDB (using cosine distance), and return each chunk’s text and metadata (doc_id, category, priority, date, chunk_idx, chunk_id, distance).

6. Ensure the retrieval output covers all requested metadata fields.

7. Build an evaluation script (recall_evaluation.py): For each query and list of relevant doc_ids, compute recall@5 by checking if at least one top-5 result comes from a relevant doc_id.

8. Test end-to-end by running ingest_and_embed.py (to re-chunk and index all data), then retrieve.py (for ad-hoc example queries and spot-checking for metadata and relevance), then recall_evaluation.py (for recall@5 metrics).

9. Optionally, tune chunk, overlap, or embedding batch size for speed/memory.

