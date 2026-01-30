import random
from pathlib import Path
from tokenizer_implementation import Tokenizer

# Paths
DATA_DIR = Path("/home/kantas/koa_scratch/ece405-assignment1-basics/data")
TOKENIZER_DIR = Path("/home/kantas/koa_scratch/ece405-assignment1-basics/data")

def load_documents(filepath, num_samples=10, seed=42):
    """Load and sample documents split by <|endoftext|>"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by the special token
    documents = content.split("<|endoftext|>")
    # Filter out empty documents
    documents = [doc.strip() for doc in documents if doc.strip()]
    
    # Random sample
    random.seed(seed)
    sampled = random.sample(documents, min(num_samples, len(documents)))
    return sampled

def calculate_compression_ratio(tokenizer, documents):
    """Calculate bytes/token compression ratio"""
    total_bytes = 0
    total_tokens = 0
    
    for doc in documents:
        text_bytes = len(doc.encode('utf-8'))
        tokens = tokenizer.encode(doc)
        total_bytes += text_bytes
        total_tokens += len(tokens)
    
    return total_bytes / total_tokens if total_tokens > 0 else 0

def main():
    # Load tokenizers
    print("Loading tokenizers...")
    tiny_tokenizer = Tokenizer.from_files(
        vocab_filepath=TOKENIZER_DIR / "vocab.json",        # TinyStories vocab
        merges_filepath=TOKENIZER_DIR / "merges.json",      # TinyStories merges
        special_tokens=["<|endoftext|>"]
    )
    
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath=TOKENIZER_DIR / "vocab_owt.json",    # OWT vocab
        merges_filepath=TOKENIZER_DIR / "merges_owt.json",  # OWT merges
        special_tokens=["<|endoftext|>"]
    )
    
    # Sample 10 documents from each validation set
    print("\nSampling 10 documents from each validation set...")
    tiny_docs = load_documents(DATA_DIR / "TinyStoriesV2-GPT4-valid.txt", num_samples=10)
    owt_docs = load_documents(DATA_DIR / "owt_valid.txt", num_samples=10)
    
    print(f"TinyStories: sampled {len(tiny_docs)} documents")
    print(f"OWT: sampled {len(owt_docs)} documents")
    
    # Part (a): Encode with matching tokenizers and calculate compression
    print("\n" + "="*60)
    print("Part (a): Compression ratios with matching tokenizers")
    print("="*60)
    
    tiny_ratio = calculate_compression_ratio(tiny_tokenizer, tiny_docs)
    owt_ratio = calculate_compression_ratio(owt_tokenizer, owt_docs)
    
    print(f"TinyStories tokenizer on TinyStories docs: {tiny_ratio:.2f} bytes/token")
    print(f"OWT tokenizer on OWT docs: {owt_ratio:.2f} bytes/token")
    
    # Part (b): Cross-tokenization (OWT docs with TinyStories tokenizer)
    print("\n" + "="*60)
    print("Part (b): OWT documents with TinyStories tokenizer")
    print("="*60)
    
    owt_with_tiny_ratio = calculate_compression_ratio(tiny_tokenizer, owt_docs)
    print(f"TinyStories tokenizer on OWT docs: {owt_with_tiny_ratio:.2f} bytes/token")
    print(f"OWT tokenizer on OWT docs: {owt_ratio:.2f} bytes/token")
    print(f"\nDifference: {owt_ratio - owt_with_tiny_ratio:.2f} bytes/token")
    
    # Show example tokenization
    print("\nExample: First OWT document snippet")
    snippet = owt_docs[0][:200] + "..." if len(owt_docs[0]) > 200 else owt_docs[0]
    print(f"Text: {snippet}")
    print(f"Tokens (TinyStories): {len(tiny_tokenizer.encode(owt_docs[0]))}")
    print(f"Tokens (OWT): {len(owt_tokenizer.encode(owt_docs[0]))}")
    
    # Part (c): Throughput estimation
    print("\n" + "="*60)
    print("Part (c): Throughput estimation")
    print("="*60)
    
    import time
    
    # Use a larger sample for throughput measurement
    test_text = "\n".join(owt_docs)
    test_bytes = len(test_text.encode('utf-8'))
    
    start = time.time()
    for _ in range(10):  # Run multiple times for accuracy
        _ = owt_tokenizer.encode(test_text)
    elapsed = time.time() - start
    
    bytes_per_second = (test_bytes * 10) / elapsed
    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * (1024**3)
    estimated_time_seconds = pile_size_bytes / bytes_per_second
    estimated_time_hours = estimated_time_seconds / 3600
    
    print(f"Throughput: {bytes_per_second / 1e6:.2f} MB/second")
    print(f"Estimated time for Pile (825GB): {estimated_time_hours:.1f} hours")


if __name__ == "__main__":
    main()