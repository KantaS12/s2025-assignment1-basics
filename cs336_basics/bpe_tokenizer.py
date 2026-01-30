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
    ordered_merges: List[Tuple[bytes, bytes]] = field(default_factory=list)

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
    # GPT-2 Standard Byte-to-Unicode Mapping
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}

    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    if special_tokens:
        escaped_specials = "|".join(re.escape(t) for t in special_tokens)
        text_segments = re.split(escaped_specials, chunk)
    else:
        text_segments = [chunk]
            
    word_counts = defaultdict(int)
    pattern = re.compile(pattern_str)
    
    for segment in text_segments:
        for match in re.finditer(pattern, segment):
            word_bytes = match.group().encode("utf-8")
            safe_word = tuple(byte_encoder[b] for b in word_bytes)
            word_counts[safe_word] += 1
        
    return dict(word_counts)

def train_bpe(
    filename: str,
    vocab_size: int = 1000,
    special_tokens: Optional[List[str]] = None,
    use_multiprocessing: bool = True
) -> BPETokenizerParams:
    # vocab_size = 256 + num_merges + num_special
    unique_specials = list(dict.fromkeys(special_tokens)) if special_tokens else []
    num_merges = vocab_size - 256 - len(unique_specials)
    
    word_counts: Dict[tuple, int] = defaultdict(int)
    split_token = b"<|endoftext|>" if "<|endoftext|>" in unique_specials else b"\n"
    
    with open(filename, "rb") as f:
        num_processes = cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    
    chunk_args = [(filename, boundaries[i], boundaries[i+1], PAT, unique_specials) for i in range(len(boundaries) - 1)]
    
    if use_multiprocessing:
        with Pool(num_processes) as pool:
            chunk_results = pool.map(process_chunk_from_file, chunk_args)
    else:
        chunk_results = [process_chunk_from_file(args) for args in chunk_args]
    
    for chunk_word_counts in chunk_results:
        for word, count in chunk_word_counts.items(): 
            word_counts[word] += count

    # Standard GPT-2 byte mapping logic
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    char_to_byte = {chr(c): b for b, c in zip(bs, cs)}
    
    vocab: Dict[int, bytes] = {b: bytes([b]) for b in range(256)}
    
    # Convert safe unicode words to integer lists
    words = [[char_to_byte[ch] for ch in w] for w in word_counts.keys()]
    word_freqs = list(word_counts.values())
    
    internal_merges: Dict[Tuple[int, int], int] = {}
    ordered_merges_bytes: List[Tuple[bytes, bytes]] = []
    
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    
    # Initial pair counting
    for word_idx, word in enumerate(words):
        freq = word_freqs[word_idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_idx)

    for merge_idx in range(num_merges):
        if not pair_counts:
            break
        
        # Max frequency, then reverse lexicographic order of vocab bytes
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))
        
        if pair_counts[best_pair] <= 0:
            break
        
        new_index = 256 + merge_idx
        internal_merges[best_pair] = new_index
        ordered_merges_bytes.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_index] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        # Apply merge and update pair counts incrementally
        for word_idx in list(pair_to_words.get(best_pair, set())):
            word = words[word_idx]
            freq = word_freqs[word_idx]
            
            # Remove old pairs from counts
            for i in range(len(word) - 1):
                old_p = (word[i], word[i+1])
                pair_counts[old_p] -= freq
                pair_to_words[old_p].discard(word_idx)
            
            # Perform merge
            new_word, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i+1]) == best_pair:
                    new_word.append(new_index)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[word_idx] = new_word
            
            # Add new pairs to counts
            for i in range(len(new_word) - 1):
                new_p = (new_word[i], new_word[i+1])
                pair_counts[new_p] += freq
                pair_to_words[new_p].add(word_idx)
        
        # Clean up zero-count pairs
        pairs_to_delete = [p for p in pair_counts if pair_counts[p] <= 0]
        for p in pairs_to_delete:
            del pair_counts[p]
            if p in pair_to_words:
                del pair_to_words[p]

    special_tokens_dict = {}
    if unique_specials:
        next_idx = 256 + len(ordered_merges_bytes)
        for t in unique_specials:
            vocab[next_idx] = t.encode("utf-8")
            special_tokens_dict[t] = next_idx
            next_idx += 1
    
    return BPETokenizerParams(
        vocab=vocab, 
        merges=internal_merges, 
        special_tokens=special_tokens_dict,
        ordered_merges=ordered_merges_bytes
    )

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
        byte_fragments = []
        for idx in indices:
            token_bytes = self.params.vocab[idx]
            byte_fragments.append(token_bytes)
        return b"".join(byte_fragments).decode("utf-8", errors="replace")

def serialize_tokenizer(params: BPETokenizerParams, vocab_file: str, merges_file: str):
    with open(vocab_file, 'w') as f: json.dump({k: list(v) for k, v in params.vocab.items()}, f)
    with open(merges_file, 'w') as f: json.dump({f"{k[0]},{k[1]}": v for k, v in params.merges.items()}, f)

if __name__ == "__main__":
    tiny_stories_train = '/home/kantas/koa_scratch/ece405-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    tiny_stories_valid = '/home/kantas/koa_scratch/ece405-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'

    owl_train = '/home/kantas/koa_scratch/ece405-assignment1-basics/data/owt_train.txt'
    owl_valid = '/home/kantas/koa_scratch/ece405-assignment1-basics/data/owt_valid.txt'

    print("\nTraining BPE Tokenizer\n")
    vocab_size = 10000
    
    profile = cProfile.Profile(); profile.enable()
    start_time, start_mem = time.time(), psutil.Process().memory_info().rss / (1024**3)

    params = train_bpe(filename=tiny_stories_train, vocab_size=vocab_size, special_tokens=["<|endoftext|>"], use_multiprocessing=True)   
    
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