from typing import Iterable, Iterator
import regex as re
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    
    def __init__(self, vocab, merges, special_tokens=None):
        """
        @param
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None

        @return
        None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else {}

        # Build helper structures
        self.token_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_priority = {merge: idx for idx, merge in enumerate(self.merges)}
        self.pattern = re.compile(PAT)
        
        # Build special token regex
        if self.special_tokens:
            self.special_regex = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)) + ")"
            )
        else:
            self.special_regex = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges. 

        @param
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None

        @return
        Tokenizer
        """
        # Load vocab from JSON format: {"index": [byte_values]}
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            raw_vocab = json.load(vf)
        
        # Convert {"index": [byte_values]} -> {int: bytes}
        vocab = {int(idx): bytes(byte_list) for idx, byte_list in raw_vocab.items()}
        
        # Load merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            merges_data = json.load(mf)
            for pair in merges_data:
                # Handle both formats
                if isinstance(pair[0], list):
                    # Format: [[byte_list1], [byte_list2]]
                    merges.append((bytes(pair[0]), bytes(pair[1])))
                else:
                    # Format: ["string1", "string2"]
                    merges.append((pair[0].encode('utf-8'), pair[1].encode('utf-8')))
        
        special_tokens_dict = {}
        if special_tokens is not None:
            # Build reverse lookup: bytes -> id
            bytes_to_id = {v: k for k, v in vocab.items()}
            next_id = max(vocab.keys()) + 1 if vocab else 0
            
            for token in special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes in bytes_to_id:
                    # Token already exists in vocab - use existing ID
                    special_tokens_dict[token] = bytes_to_id[token_bytes]
                else:
                    # Add new special token
                    special_tokens_dict[token] = next_id
                    vocab[next_id] = token_bytes
                    next_id += 1

        return cls(vocab, merges, special_tokens_dict)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        @param
        self: self
        text: str

        @return
        encoded: list[int]
        """

        # Handle special tokens
        if self.special_regex:
            segments = self.special_regex.split(text)
        else:
            segments = [text]
        
        result = []
        for segment in segments:
            if segment in self.special_tokens:
                result.append(self.special_tokens[segment])
            elif segment:
                # Pre-tokenize and encode each word
                for match in re.finditer(self.pattern, segment):
                    word = [bytes([b]) for b in match.group().encode("utf-8")]
                    
                    # Apply merges by priority
                    while len(word) > 1:
                        best_pair, best_priority, best_idx = None, float('inf'), -1
                        for i in range(len(word) - 1):
                            pair = (word[i], word[i + 1])
                            if pair in self.merge_priority and self.merge_priority[pair] < best_priority:
                                best_priority = self.merge_priority[pair]
                                best_pair = pair
                                best_idx = i
                        if best_pair is None:
                            break
                        word = word[:best_idx] + [best_pair[0] + best_pair[1]] + word[best_idx + 2:]
                    
                    # Convert tokens to IDs with fallback
                    for token in word:
                        if token in self.token_to_id:
                            result.append(self.token_to_id[token])
                        else:
                            # Fallback: split into individual bytes (IDs 0-255 always exist)
                            for b in token:
                                result.append(b)
        
        return result

        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given a iterable of strings, return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.

        @param
        self: self
        iterable: Iterable[str]

        @return
        generator: Iterator[int]
        """
        
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

        

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        @param
        self: self
        ids: list[int]

        @return
        texts: str
        """

        id_to_special = {v: k for k, v in self.special_tokens.items()}
        result = []
        for id in ids:
            if id in id_to_special:
                result.append(id_to_special[id].encode('utf-8'))
            elif id in self.vocab:
                result.append(self.vocab[id])
        return b''.join(result).decode('utf-8', errors='replace')