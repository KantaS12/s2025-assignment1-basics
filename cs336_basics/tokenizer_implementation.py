


class Tokenizer:
    
    def __init__(self, vocab, merges, special_tokens=None){
        """
        @param
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None

        @return
        None
        """
        pass
    }

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None){
        """
        @param
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None

        @return
        Tokenizer
        """
        pass
    }

    def encode(self, text: str) -> list[int]{
        """
        @param
        self: self
        text: str

        @return
        encoded: list[int]
        """
        pass
    }

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]{
        """
        @param
        self: self
        iterable: Iterable[str]

        @return
        generator: Iterator[int]
        """
        pass
    }

    def decode(self, ids: list[int]) -> str{
        """
        @param
        self: self
        ids: list[int]

        @return
        texts: str
        """
        pass
    }