from typing import List, Union
from pathlib import Path
from tqdm import tqdm
from ..models.qa_pair import Chunk
from ..converters.text import TextConverter
from ..strategies.chunk import ChunkStrategy

class ChunkGenerator:
    def __init__(
            self,
            text_converter: TextConverter,
            chunk_strategy: ChunkStrategy,
    ):
        self.text_converter = text_converter
        self.chunk_strategy = chunk_strategy
    
    def generate(self, input_path: Union[str, Path]) -> List[Chunk]:
        '''
        file/dir -> chunks
        handling both file or directory path
        '''
        chunks = []
        input_path = Path(input_path)

        files = [input_path] if input_path.is_file() else [path for path in input_path.glob('**/*') if path.is_file()]
        
        for file in tqdm(files, desc="Extract Content"):
            text = self.text_converter.convert(file)
            chunks.extend(self.chunk_strategy.split(source=file, text=text))
        
        return chunks





