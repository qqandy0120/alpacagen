from .main import AlpacaGen
from .models.qa_pair import QAPair, Chunk
from .strategies.chunk import ChunkStrategy, RecursiveChunkStrategy
from .converters.text import TextConverter, MarkItDownConverter

__version__ = "0.1.0"

__all__ = [
    'AlpacaGen',
    'QAPair',
    'Chunk',
    'ChunkStrategy',
    'RecursiveChunkStrategy',
    'TextConverter',
    'MarkItDownConverter',
]