import nltk
import tiktoken

nltk.download('punkt', quiet=True)

def tokenize_text(text, encoding_name="gpt2"):
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)

def detokenize_tokens(tokens, encoding_name="gpt2"):
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.decode(tokens)

def chunk_text(text, chunk_size=200, overlap=50, encoding_name="gpt2"):
    tokens = tokenize_text(text, encoding_name)
    chunks = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = tokens[start:end]
        chunks.append(detokenize_tokens(chunk, encoding_name))
        if end == n:
            break
        start += chunk_size - overlap
    return chunks
