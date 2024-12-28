import asyncio
from typing import List, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from ..models.qa_pair import QAPair, Chunk
from ..converters.text import TextConverter
from ..strategies.chunk import ChunkStrategy
from .qa import QAGenerator

class QADatasetGenerator:
    def __init__(
        self,
        text_converter: TextConverter,
        chunk_strategy: ChunkStrategy,
        qa_generator: QAGenerator,
        entries_per_chunk: int = 3
    ):
        self.text_converter = text_converter
        self.chunk_strategy = chunk_strategy
        self.qa_generator = qa_generator
        self.entries_per_chunk = entries_per_chunk
    
    async def process_file(self, input_path: Union[str, Path]) -> Tuple[List[Chunk], List[QAPair]]:
        text = self.text_converter.convert(input_path)
        chunks: List[Chunk] = self.chunk_strategy.split(source=input_path, text=text)
        
        tasks = []
        for chunk in chunks:
            tasks.extend([self.qa_generator.generate(chunk.content) 
                         for _ in range(self.entries_per_chunk)])
        
        results = await tqdm_asyncio.gather(*tasks, desc="Generate QA")
        dataset: List[QAPair] = [result for result in results if result is not None]
        
        return chunks, dataset
    
    async def process_directory(self, input_dir: Union[str, Path]) -> Tuple[List[Chunk], List[QAPair]]:
        BATCH_SIZE = 20
        input_dir = Path(input_dir)
        chunks = []
        tasks = []
        dataset = []
        
        paths = list(input_dir.glob('**/*'))
        files = [path for path in paths if path.is_file()]

        for input_path in tqdm(files, desc="Extract Content"):
            if input_path.is_file():
                text = self.text_converter.convert(input_path)
                chunks.extend(self.chunk_strategy.split(source=input_path, text=text))
        
        for chunk in chunks:
            tasks.extend([self.qa_generator.generate(chunk.content) 
                         for _ in range(self.entries_per_chunk)])

        if not tasks:
            return chunks, []

        itrs = (len(tasks) // BATCH_SIZE) + (len(tasks) % BATCH_SIZE != 0)
        print(f"Total QAPairs: {len(tasks)}")
        
        for n, ith in enumerate(range(0, len(tasks), BATCH_SIZE)):
            batch_tasks = tasks[ith: ith + BATCH_SIZE]
            batch_dataset = await tqdm_asyncio.gather(
                *batch_tasks, 
                desc=f"Generate QA(Batch{n+1}/{itrs})"
            )
            dataset.extend(batch_dataset)
            
            del batch_tasks
            del batch_dataset
            await asyncio.sleep(0.5)
        
        dataset = [result for result in dataset if result is not None]
        return chunks, dataset