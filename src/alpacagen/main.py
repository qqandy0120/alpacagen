import json
import logging
import asyncio
from typing import List, Optional, Union, Tuple, Any
from pathlib import Path
import nest_asyncio
from openai import AsyncOpenAI

from .converters.text import MarkItDownConverter
from .models.qa_pair import QAPair, Chunk
from .strategies.chunk import RecursiveChunkStrategy
from .generators.chunk import ChunkGenerator
from .generators.qa import QAGenerator
from .generators.dataset import QADatasetGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LANGUAGES = ['zhtw', 'en']
LLM_CLIENTS = ['azure', 'openai']
DEFAULT_MODEL_DICT = {
    'azure': 'azure-gpt-4o',
    'openai': 'gpt-4o',
}
DEFAULT_PROMPT_PATHS = {
    lang: Path(__file__).parent / "prompt" / f"prompt_gen_{lang}.txt" 
    for lang in LANGUAGES
}

class AlpacaGen:
    def __init__(
            self,
            llm_client: str = None,
            llm_model: str = None,
            api_key: str = None,
            base_url: str = None,
    ):
        assert llm_client in LLM_CLIENTS, f"Specify your llm client, client should be one of {LLM_CLIENTS}"
        
        self.llm_client = llm_client
        self.llm_model = llm_model if llm_model else DEFAULT_MODEL_DICT[self.llm_client]
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
        logging.basicConfig(level=logging.ERROR)

    async def _get_client(self) -> AsyncOpenAI:
        """Get or create AsyncOpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def get_chunks(
        self,
        input_path: Union[str, Path],
        chunk_size: int = 4096,
    ) -> List[Chunk]:
        """Generate chunks from input file."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Cannot find: {input_path}")

        chunk_generator = ChunkGenerator(
            text_converter=MarkItDownConverter(),
            chunk_strategy=RecursiveChunkStrategy(
                chunk_size=chunk_size,
                chunk_overlap=200,
            )
        )

        return chunk_generator.generate(input_path)

    async def _get_dataset_async(
        self,
        chunks: List[Chunk],
        language: str = 'zhtw',
        gen_prompt_path: Optional[Path] = None,
        entries_per_chunk: int = 3,
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[QAPair]:
        """Internal async method for dataset generation."""
        assert language in LANGUAGES, f"Language should be one of {LANGUAGES}"

        def get_default_prompt(language: str) -> str:
            with DEFAULT_PROMPT_PATHS[language].open('r', encoding='utf-8') as f:
                return f.read()

        gen_prompt = (
            Path(gen_prompt_path).read_text(encoding='utf-8')
            if gen_prompt_path
            else get_default_prompt(language)
        )

        client = await self._get_client()
        
        dataset_generator = QADatasetGenerator(
            qa_generator=QAGenerator(
                client,
                self.llm_model,
                gen_prompt
            ),
            entries_per_chunk=entries_per_chunk
        )

        dataset = await dataset_generator.generate(chunks)

        if output_path:
            self.save_to_jsonl(dataset, output_path)

        return dataset

    def get_datasets(
        self,
        chunks: List[Chunk],
        language: str = 'zhtw',
        gen_prompt_path: Optional[Path] = None,
        entries_per_chunk: int = 3,
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[QAPair]:
        """
        User-friendly synchronous method to generate datasets from chunks.
        
        Args:
            chunks: List of text chunks to process
            language: Language for prompt generation ('zhtw' or 'en')
            gen_prompt_path: Optional custom prompt file path
            entries_per_chunk: Number of QA pairs to generate per chunk
            output_path: Optional path to save the dataset
            
        Returns:
            List of QAPair objects representing the generated dataset
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                nest_asyncio.apply()
            return loop.run_until_complete(
                self._get_dataset_async(
                    chunks,
                    language,
                    gen_prompt_path,
                    entries_per_chunk,
                    output_path
                )
            )
        except RuntimeError:
            return asyncio.run(
                self._get_dataset_async(
                    chunks,
                    language,
                    gen_prompt_path,
                    entries_per_chunk,
                    output_path
                )
            )

    def save_to_jsonl(self, dataset: List[QAPair], output_path: Union[str, Path]):
        """Save dataset (list of QAPair) to a JSONL file."""
        try:
            with Path(output_path).open('w', encoding='utf-8') as f:
                for qa_pair in dataset:
                    json_line = json.dumps(qa_pair.to_dict(), ensure_ascii=False)
                    f.write(json_line + '\n')
        except IOError as e:
            logger.error(f"Error writing to file {output_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            raise