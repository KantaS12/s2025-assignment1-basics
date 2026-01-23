from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, BinaryIO
from collections import defaultdict
import regex as re
import cProfile
import psutil
from io import StringIO
from multiprocessing import Pool, cpu_count
import os
import json
import time
import pstats
import heapq

@dataclass(frozen=True)
class BPETokenizerParams:
    """Data container for a trained BPETokenizer."""
    vocab: Dict[int, bytes]             # index -> bytes
    merges: Dict[Tuple[int, int], int]  # (index1, index2) -> new_index
    special_tokens: Dict[str, int] = field(default_factory=dict)

# Optimized regex for GPT-2/3 style pre-tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Finds byte offsets in a file to split work across CPU cores."""
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def process_chunk_from_file(args):
    """Worker function to process a slice of the text file into word counts."""
    filename, start, end, pattern_str, special_tokens = args
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    if special_tokens:
        for token in special_tokens:
            chunk = chunk.replace(token, "")
    pattern = re.compile(pattern_str)
    word_counts: Dict[tuple, int] = defaultdict(int)
    for match in re.finditer(pattern, chunk):
        word_bytes = tuple(map(int, match.group().encode("utf-8")))
        word_counts[word_bytes] += 1
    return dict(word_counts)

def train_bpe(
    filename: str,
    num_merges: int = 1000,
    special_tokens: Optional[List[str]] = None,
    use_multiprocessing: bool = True
) -> BPETokenizerParams:
    """Pure BPE training logic using a Max-Heap for efficiency."""
    
    # Word counting
    word_counts: Dict[tuple, int] = defaultdict(int)
    split_token = b"<|endoftext|>" if special_tokens and "<|endoftext|>" in special_tokens else b"\n"
    
    print(f"Reading {filename} and counting words...")
    with open(filename, "rb") as f:
        num_processes = cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    
    chunk_args = [(filename, boundaries[i], boundaries[i+1], PAT, special_tokens) for i in range(len(boundaries) - 1)]
    with Pool(num_processes) as pool:
        chunk_results = pool.map(process_chunk_from_file, chunk_args)
    for chunk_word_counts in chunk_results:
        for word, count in chunk_word_counts.items(): word_counts[word] += count

    # Initialization
    words = [list(w) for w in word_counts.keys()]
    word_freqs = list(word_counts.values())
    vocab: Dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges: Dict[Tuple[int, int], int] = {}
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    
    # Populating initial pair counts
    for word_idx, word in enumerate(words):
        freq = word_freqs[word_idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_idx)

    # Initialize Max Heap
    pq = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(pq)
    
    print(f"Starting merge loop for {num_merges} merges...")
    
    # Main merge 
    for merge_idx in range(num_merges):
        # Retrieve the most frequent pair from the heap
        while pq:
            neg_count, best_pair = heapq.heappop(pq)
            if pair_counts.get(best_pair, 0) == -neg_count:
                break
        else: break
        
        if pair_counts[best_pair] <= 0: break
        
        # Define new token
        new_index = 256 + merge_idx
        merges[best_pair] = new_index
        vocab[new_index] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        # Update only the words that contain the winning pair
        words_to_update = list(pair_to_words.get(best_pair, set()))
        for word_idx in words_to_update:
            word, freq = words[word_idx], word_freqs[word_idx]
            
            # Remove old pair counts from the dictionary
            for i in range(len(word) - 1):
                old_p = (word[i], word[i+1])
                pair_counts[old_p] -= freq
                pair_to_words[old_p].discard(word_idx)
            
            # Apply the merge
            new_word, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i+1]) == best_pair:
                    new_word.append(new_index); i += 2
                else:
                    new_word.append(word[i]); i += 1
            words[word_idx] = new_word
            
            # Add new pair counts generated by this merge
            for i in range(len(new_word) - 1):
                new_p = (new_word[i], new_word[i+1])
                pair_counts[new_p] += freq
                pair_to_words[new_p].add(word_idx)
                # Lazy push to heap
                heapq.heappush(pq, (-pair_counts[new_p], new_p))
        
        # Clean up the winning pair
        pair_counts.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)
        
        if (merge_idx + 1) % 1000 == 0:
            print(f"Completed merge {merge_idx + 1}/{num_merges}")

    # Final
    special_tokens_dict = {}
    if special_tokens:
        next_idx = 256 + len(merges)
        for t in special_tokens:
            vocab[next_idx] = t.encode("utf-8")
            special_tokens_dict[t] = next_idx
            next_idx += 1
    
    print(f"Unique words processed: {len(word_counts):,}")
    return BPETokenizerParams(vocab=vocab, merges=merges, special_tokens=special_tokens_dict)

class BPETokenizer:
    def __init__(self, params: BPETokenizerParams):
        self.params, self.pattern = params, re.compile(PAT)
        self.merge_priority = {pair: idx for idx, pair in enumerate(params.merges.keys())}
        self.special_regex = re.compile(f"({'|'.join(map(re.escape, params.special_tokens.keys()))})") if params.special_tokens else None

    def encode(self, text: str) -> List[int]:
        if not self.special_regex: return self._encode_standard_text(text)
        indices = []
        for part in self.special_regex.split(text):
            if part in self.params.special_tokens: indices.append(self.params.special_tokens[part])
            elif part: indices.extend(self._encode_standard_text(part))
        return indices

    def _encode_standard_text(self, text: str) -> List[int]:
        indices = []
        for match in re.finditer(self.pattern, text):
            chunk_indices = list(match.group().encode("utf-8"))
            while len(chunk_indices) > 1:
                best_pair, best_priority, best_idx = None, float('inf'), -1
                for i in range(len(chunk_indices) - 1):
                    pair = (chunk_indices[i], chunk_indices[i+1])
                    if pair in self.merge_priority and self.merge_priority[pair] < best_priority:
                        best_priority, best_pair, best_idx = self.merge_priority[pair], pair, i
                if best_pair is None: break
                chunk_indices = chunk_indices[:best_idx] + [self.params.merges[best_pair]] + chunk_indices[best_idx+2:]
            indices.extend(chunk_indices)
        return indices

    def decode(self, indices: List[int]) -> str:
        return b"".join(self.params.vocab[idx] for idx in indices).decode("utf-8", errors="replace")

def serialize_tokenizer(params: BPETokenizerParams, vocab_file: str, merges_file: str):
    with open(vocab_file, 'w') as f: json.dump({k: list(v) for k, v in params.vocab.items()}, f)
    with open(merges_file, 'w') as f: json.dump({f"{k[0]},{k[1]}": v for k, v in params.merges.items()}, f)

if __name__ == "__main__":
    tiny_stories_train = 'data/TinyStoriesV2-GPT4-train.txt'
    tiny_stories_valid = 'data/TinyStoriesV2-GPT4-valid.txt'

    owl_train = 'data/owt_train.txt'
    owl_valid = 'data/owt_valid.txt'

    """
    print("\nTraining BPE Tokenizer\n")
    vocab_size = 10000
    num_merges = vocab_size - 256 - 1
    
    profile = cProfile.Profile(); profile.enable()
    start_time, start_mem = time.time(), psutil.Process().memory_info().rss / (1024**3)

    params = train_bpe(filename=tiny_stories_train, num_merges=num_merges, special_tokens=["<|endoftext|>"], use_multiprocessing=True)   
    
    end_time, end_mem = time.time(), psutil.Process().memory_info().rss / (1024**3)
    profile.disable()

    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Peak Memory usage: {end_mem - start_mem:.4f} GB")
    serialize_tokenizer(params, "vocab.json", "merges.json")
    
    longest_token_bytes = params.vocab[max(params.vocab.keys(), key=lambda k: len(params.vocab[k]))]
    print(f"Longest token: {longest_token_bytes} ({len(longest_token_bytes)} bytes)")

    s = StringIO(); pstats.Stats(profile, stream=s).sort_stats('tottime').print_stats(1); print(s.getvalue())
    
    print("\nTesting Tokenizer on Validation Set\n")
    tokenizer = BPETokenizer(params)
    with open(tiny_stories_valid, 'r', encoding='utf-8') as f: valid_text = f.read()
    
    v_tokens = tokenizer.encode(valid_text)
    print(f"Validation encoded tokens: {len(v_tokens)}")
    print(f"Compression ratio: {len(valid_text.encode('utf-8')) / len(v_tokens):.2f}x")
    print(f"Decoded matches original: {tokenizer.decode(v_tokens) == valid_text}")
    """

    print("\nTraining BPE Tokenizer on OWT Dataset\n")
    vocab_size = 32000
    num_merges = vocab_size - 256 - 1

    profile = cProfile.Profile(); profile.enable()
    start_time, start_mem = time.time(), psutil.Process().memory_info().rss / (1024**3)

    params = train_bpe(filename=owl_train, num_merges=num_merges, special_tokens=["<|endoftext|>"], use_multiprocessing=True)   

    end_time, end_mem = time.time(), psutil.Process().memory_info().rss / (1024**3)
    profile.disable()

    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Peak Memory usage: {end_mem - start_mem:.4f} GB")

    serialize_tokenizer(params, "vocab_owt.json", "merges_owt.json")
    longest_token_bytes = params.vocab[max(params.vocab.keys(), key=lambda k: len(params.vocab[k]))]
    print(f"Longest token: {longest_token_bytes} ({len(longest_token_bytes)} bytes)")
    
    s = StringIO(); pstats.Stats(profile, stream=s).sort_stats('tottime').print_stats(1); print(s.getvalue())

    print("\nTesting Tokenizer on OWT Validation Set\n")
    tokenizer = BPETokenizer(params)
    with open(owl_valid, 'r', encoding='utf-8') as f: valid_text = f.read()
    v_tokens = tokenizer.encode(valid_text)
    
    print(f"Validation encoded tokens: {len(v_tokens)}")
    print(f"Compression ratio: {len(valid_text.encode('utf-8')) / len(v_tokens):.2f}x")
    print(f"Decoded matches original: {tokenizer.decode(v_tokens) == valid_text}")