from typing import List, Union
from pathlib import Path
from abc import ABC, abstractmethod
from models.qa_pair import Chunk
class ChunkStrategy(ABC):
    @abstractmethod
    def _split_text(self, text: str) -> List[str]:
        # customize in each strategy
        pass

    def split(self, source: Union[str, Path], text: str) -> List[Chunk]:
        chunks = self._split_text(text)
        return [
            Chunk(
                content=chunk,
                source=source,
                idx=f"{str(n+1).zfill(len(str(len(chunks))))}/{len(chunks)}"
            ) 
            for n, chunk in enumerate(chunks)
        ]

class RecursiveChunkStrategy(ChunkStrategy):
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def _split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)