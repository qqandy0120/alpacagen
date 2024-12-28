# AlpacaGen

AlpacaGen is a powerful tool for generating instruction-following datasets in the Alpaca format using large language models (LLMs). It can process both single files and entire directories, automatically chunking content and generating multiple instruction-input-output pairs for each chunk.

## Features

- Support for multiple LLM providers (Azure OpenAI, OpenAI)
- Multilingual support (Traditional Chinese, English)
- Automatic text chunking with customizable size and overlap
- Batch processing with progress bars
- Configurable number of QA pairs per chunk
- JSONL output format

## Installation

```bash
pip install markitdown langchain_text_splitters openai tqdm nest_asyncio
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

# Generate dataset from a single file
chunks, dataset = ag.generate(
    input_path='your_file.txt',
    output_path='output.jsonl',
    language='zhtw',  # or 'en'
    entries_per_chunk=3
)
```

### Advanced Configuration

```python
# Process an entire directory with custom settings
chunks, dataset = ag.generate(
    input_path='your_directory/',
    output_path='output.jsonl',
    language='en',
    gen_prompt_path='custom_prompt.txt',  # Optional: Use custom prompt template
    chunk_size=4096,  # Customize chunk size
    entries_per_chunk=5  # Generate more QA pairs per chunk
)
```

## Output Format

The generated JSONL file contains entries in the following format:

```json
{
    "instruction": "Task description",
    "input": "Additional context or input",
    "output": "Expected response",
    "text": "Full formatted text including all components"
}
```

## Customization

### Custom Prompts

Create a text file with your prompt template. Use `{text}` as a placeholder for the chunk content:

```text
Based on the following text, generate a question-answer pair in JSON format:
{text}
Generate a response in the following format:
{
    "instruction": "The task or question",
    "input": "Any additional context or input",
    "output": "The expected response or answer"
}
```

### Configuration Options

- `llm_client`: Choose between 'azure' or 'openai'
- `llm_model`: Specify custom model (defaults available for each client)
- `chunk_size`: Control the size of text chunks (default: 4096)
- `entries_per_chunk`: Number of QA pairs to generate per chunk (default: 3)
- `language`: Choose between 'zhtw' (Traditional Chinese) or 'en' (English)

## Error Handling

AlpacaGen includes comprehensive error handling and logging:
- File processing errors are logged
- Invalid JSON responses are skipped
- Progress bars show processing status
- Failed generations are automatically filtered out

## Understanding Chunks and Dataset

### Chunks

Chunks are sections of your input text that have been automatically split for processing. Each chunk contains:
- `content`: The actual text content
- `source`: The source file path
- `idx`: A formatted string showing the chunk's position (e.g., "01/17" means chunk 1 of 17)

Example chunks:
```python
# Example chunks from a technical document
[
    Chunk(
        content="Introduction to Machine Learning\nMachine learning is a subset of artificial intelligence...",
        source="ml_guide.txt",
        idx="01/03"
    ),
    Chunk(
        content="Supervised Learning Methods\nIn supervised learning, algorithms learn from labeled data...",
        source="ml_guide.txt",
        idx="02/03"
    ),
    Chunk(
        content="Practical Applications\nMachine learning is used in various fields including...",
        source="ml_guide.txt",
        idx="03/03"
    )
]
```

### Dataset (QA Pairs)

The dataset consists of QA pairs generated from each chunk. Each QA pair contains:
- `instruction`: The task or question
- `input`: Additional context or input data
- `output`: The expected response or answer

Example QA pairs:
```python
[
    QAPair(
        instruction="Explain the basic concept of machine learning in simple terms",
        input="Consider the following introduction to machine learning",
        output="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It's similar to how humans learn from experience, but using data and algorithms instead."
    ),
    QAPair(
        instruction="What is the main characteristic of supervised learning?",
        input="Read about supervised learning methods",
        output="The main characteristic of supervised learning is that it uses labeled data for training. This means the algorithm learns from examples where the correct answers are already known, allowing it to make predictions on new, unseen data."
    ),
    QAPair(
        instruction="List three practical applications of machine learning",
        input="Based on the section about practical applications",
        output="Three practical applications of machine learning include: 1) Email spam filtering, 2) Medical diagnosis and image analysis, and 3) Recommendation systems in e-commerce platforms. These applications demonstrate how machine learning can solve real-world problems."
    )
]
```

### Processing Flow

1. Input text → Chunks:
   - Text is split into manageable chunks using RecursiveChunkStrategy
   - Overlap ensures context continuity between chunks
   - Each chunk is tracked with its source and position

2. Chunks → Dataset:
   - Multiple QA pairs are generated for each chunk
   - Each QA pair focuses on different aspects of the chunk's content
   - Invalid or failed generations are automatically filtered out

3. Dataset → JSONL:
   - QA pairs are converted to JSON format
   - Each pair is written as a separate line in the output file
   - The full text including instruction, input, and output is preserved

## Best Practices

1. Start with a small test file before processing large directories
2. Monitor the generated output quality
3. Adjust chunk size based on your content
4. Use appropriate language setting for your source material
5. Consider using custom prompts for specific use cases

## Limitations

- Requires valid API credentials for OpenAI or Azure OpenAI
- Processing speed depends on API rate limits
- Large directories may take significant time to process
- Memory usage scales with chunk size and batch size