import asyncio
from typing import List
from tqdm.asyncio import tqdm_asyncio
from ..models.qa_pair import QAPair, Chunk
from .qa import QAGenerator

class QADatasetGenerator:
    def __init__(
        self,
        qa_generator: QAGenerator,
        entries_per_chunk: int = 3
    ):
        self.qa_generator = qa_generator
        self.entries_per_chunk = entries_per_chunk

    async def generate(self, chunks: List[Chunk]) -> List[QAPair]:
        BATCH_SIZE = 20
        tasks = []
        dataset = []
        for chunk in chunks:
            tasks.extend([self.qa_generator.generate(chunk) for _ in range(self.entries_per_chunk)])

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
        return dataset