import json
import logging
import asyncio
from typing import List, Optional, Union, Tuple
from pathlib import Path
import nest_asyncio
from openai import AsyncOpenAI

from .converters.text import MarkItDownConverter
from .models.qa_pair import QAPair, Chunk
from .strategies.chunk import RecursiveChunkStrategy
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
        logging.basicConfig(level=logging.ERROR)

    def generate(
        self, 
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        language: str = 'zhtw',
        gen_prompt_path: Optional[Path] = None,
        chunk_size: int = 4096,
        entries_per_chunk: int = 3
    ) -> Tuple[List[Chunk], List[QAPair]]:
        """
        Generate instruction-following dataset from input file or directory.
        
        Args:
            input_path: Path to input file or directory
            output_path: Path to output JSONL file
            language: Language for prompt template ('zhtw' or 'en')
            gen_prompt_path: Optional custom prompt template path
            chunk_size: Size of text chunks for processing
            entries_per_chunk: Number of QA pairs to generate per chunk
            
        Returns:
            Tuple of (chunks, dataset) where chunks is List[Chunk] and dataset is List[QAPair]
        """
        assert language in LANGUAGES, f"Language should be one of {LANGUAGES}"

        async def _async_generate(
            input_path, 
            output_path, 
            language, 
            gen_prompt_path, 
            chunk_size, 
            entries_per_chunk
        ) -> Tuple[List[Chunk], List[QAPair]]:
            def get_default_prompt(language: str) -> str:
                with DEFAULT_PROMPT_PATHS[language].open('r', encoding='utf-8') as f:
                    return f.read()
                
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Cannot find: {input_path}")

            gen_prompt = (
                Path(gen_prompt_path).read_text(encoding='utf-8') 
                if gen_prompt_path 
                else get_default_prompt(language)
            )

            async with AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            ) as client:
                chunk_strategy = RecursiveChunkStrategy(
                    chunk_size=chunk_size, 
                    chunk_overlap=200
                )
                qa_generator = QAGenerator(client, self.llm_model, gen_prompt)
                generator = QADatasetGenerator(
                    text_converter=MarkItDownConverter(),
                    chunk_strategy=chunk_strategy,
                    qa_generator=qa_generator,
                    entries_per_chunk=entries_per_chunk,
                )
                
                if input_path.is_dir():
                    chunks, dataset = await generator.process_directory(input_path)
                else:
                    chunks, dataset = await generator.process_file(input_path)

                if not output_path:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"{input_path.stem}_{timestamp}.jsonl"
    
                self.save_to_jsonl(dataset, output_path)
                return chunks, dataset

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                nest_asyncio.apply()
            return loop.run_until_complete(
                _async_generate(
                    input_path, 
                    output_path, 
                    language, 
                    gen_prompt_path, 
                    chunk_size, 
                    entries_per_chunk
                )
            )
        except RuntimeError:
            return asyncio.run(
                _async_generate(
                    input_path, 
                    output_path, 
                    language, 
                    gen_prompt_path, 
                    chunk_size, 
                    entries_per_chunk
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