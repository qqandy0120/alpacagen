# AlpacaGen

AlpacaGen is a powerful tool for generating instruction-following datasets in the Alpaca format using large language models (LLMs). It supports processing of various file types (PDF, DOCX, TXT, etc.) and can handle both individual files and entire directories, automatically chunking content and generating multiple instruction-input-output pairs for each chunk.

## Features

- Process multiple file formats (PDF, DOCX, TXT, and more)
- Recursive directory processing - automatically handles all supported files in a directory
- Support for multiple LLM providers (Azure OpenAI, OpenAI)
- Multilingual support (Traditional Chinese, English)
- Automatic text chunking with customizable size and overlap
- Batch processing with progress bars
- Source tracking for generated QA pairs
- JSONL output format

## Installation

```bash
pip install alpacagen
```

## Usage

### Basic Usage

```python
from alpacagen import AlpacaGen

# Initialize AlpacaGen
ag = AlpacaGen(
    llm_client='azure',  # or 'openai'
    api_key='your-api-key',
    base_url='your-api-base-url'  # Required for Azure OpenAI
)

# Process a single file
chunks = ag.get_chunks('documentation.pdf')

# Or process an entire directory
chunks = ag.get_chunks('path/to/your/docs/')  # Will process all supported files in the directory

# Generate dataset from chunks
dataset = ag.get_datasets(
    chunks,
    language='zhtw',  # or 'en'
    entries_per_chunk=3,
    output_path='output.jsonl'
)
```

### Processing Different File Types

AlpacaGen can handle various file formats:

```python
# PDF documents
chunks = ag.get_chunks('technical_manual.pdf')

# Word documents
chunks = ag.get_chunks('specifications.docx')

# Text files
chunks = ag.get_chunks('notes.txt')

# Mixed directory
chunks = ag.get_chunks(
    'project_docs/',  # Contains PDFs, DOCXs, TXTs
    chunk_size=4096
)
```

### Advanced Configuration

```python
# Process with custom settings
ag = AlpacaGen(
    llm_client='azure',
    api_key='your-api-key',
    base_url='your-api-base-url',
    llm_model='your-model-selection'  # default is gpt-4o
)

# Generate chunks with custom size
chunks = ag.get_chunks(
    input_path='docs_directory/',  # Can be directory or single file
    chunk_size=4096
)

# Generate dataset with custom prompt
dataset = ag.get_datasets(
    chunks,
    language='en',
    gen_prompt_path='custom_prompt.txt',  # Optional: Use custom prompt template
    entries_per_chunk=5,  # Generate more QA pairs per chunk
    output_path='output.jsonl'
)
```

## Understanding Data Structures

### Chunks

Chunks represent sections of your input text that have been automatically split for processing. Each chunk contains:
- `content`: The actual text content
- `source`: The source file path (preserves original file path for directory processing)
- `idx`: A formatted string showing the chunk's position (e.g., "01/17" means chunk 1 of 17)

Example chunk structure:
```python
@dataclass
class Chunk:
    content: str
    source: str  # e.g., "project_docs/specifications.docx"
    idx: str     # format: "01/17"
```

### QA Pairs

The dataset consists of QA pairs generated from each chunk. Each QA pair contains:
- `instruction`: The task or question
- `input`: Additional context or input data
- `output`: The expected response or answer
- `source`: Reference to the original chunk that generated this pair

Example QA pair structure:
```python
@dataclass
class QAPair:
    instruction: str
    input: str
    output: str
    source: Chunk  # Maintains link to original file and position
```

Example dataset output:
```python
[
    QAPair(
        instruction="What are the key components of a distributed system?",
        input="Based on the chapter about distributed systems",
        output="The key components include nodes, communication networks, middleware, and coordination mechanisms.",
        source=Chunk(
            content="Chapter 1: Distributed Systems\nA distributed system consists of...",
            source="project_docs/distributed_systems.pdf",
            idx="01/12"
        )
    )
]
```

## Configuration Options

- `llm_client`: Choose between 'azure' or 'openai'
- `llm_model`: Specify custom model (defaults available for each client)
- `chunk_size`: Control the size of text chunks (default: 4096)
- `entries_per_chunk`: Number of QA pairs to generate per chunk (default: 3)
- `language`: Choose between 'zhtw' (Traditional Chinese) or 'en' (English)

## Best Practices

1. Start with a small test file before processing large directories
2. Organize your input files in a structured directory
3. Monitor the generated output quality
4. Adjust chunk size based on your content type and length
5. Use appropriate language setting for your source material
6. Consider using custom prompts for specific domains or file types
7. Keep track of source chunks for better data traceability

## Recent Updates

- Added support for processing entire directories recursively
- Enhanced file type support (PDF, DOCX, TXT, etc.)
- Added source tracking for QA pairs through the new `source` attribute
- Updated chunking mechanism with improved overlap handling
- Enhanced dataset generation with better error handling
- Simplified API with separate `get_chunks` and `get_datasets` methods

## Contact Me

Have questions or suggestions? Feel free to reach out!

Email: [qqandy0120@gmail.com](mailto:qqandy0120@gmail.com)