from typing import Iterable, Iterator


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
        vocab = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            for line in vf:
                token, index = line.strip().split()
                vocab[int(index)] = token.encode('utf-8')
        
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            for line in mf:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))
        
        special_tokens_dict = {}
        if special_tokens is not None:
            for token in special_tokens:
                special_tokens_dict[token] = len(vocab) + len(special_tokens_dict)
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens_dict)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        @param
        self: self
        text: str

        @return
        encoded: list[int]
        """
        
        tokens = list(text.encode('utf-8'))

        return tokens

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

        bytes_list = [self.vocab[id] for id in ids if id in self.vocab]
        return b''.join(bytes_list).decode('utf-8', errors='ignore')