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
    llm_model='your-model-selection'  # defult using gpt-4o
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

### Defult Prompt
```
Task Description: You are a professional educational content developer. I will provide you with a passage from a technical manual or documentation. Based on this content, generate 1 entry for fine-tuning large language models. This entry should be directly related to the given text content and pose relevant questions.

Related materials or book content: {text}

Please generate the entry in the following format, ensuring all fields have appropriate content:
{
    "instruction": "Ask a specific question related to the text",
    "input": "Provide additional context information here if needed, otherwise leave empty",
    "output": "Detailed answer to the instruction or task completion result"
}

Examples of generated questions:
{
    "instruction": "Which server can provide 4 Intel Xeon Scalable processors Gold 6 or Intel Xeon Platinum series per server, with each processor having 8 or more cores, 3.7GHz or higher, equipped with 22.5MB or more L3 cache memory, and each CPU supporting 2 or more high-speed UPI (Ultra Path Interconnect) system buses, including CPU cooling device?",
    "input": "",
    "output": "HPE ProLiant DL560 Gen11"
}
{
    "instruction": "Which server management control chip service or server management software service (must provide legal authorization) has early warning functionality and Call Home feature, and provides proactive support services, where the server manufacturer can remotely assist in troubleshooting, and must complete this function's setup or integration with the hospital's HPE server host group's ONEVIEW automatic fault reporting mechanism or DELL open manager automatic fault reporting mechanism during installation?",
    "input": "",
    "output": "HPE Integrated Lights-Out"
}

Please generate content in "valid JSON format." Ensure that the generated questions are clear and specific, avoid using pronouns to refer to content in the data, and use correct and complete nouns or concepts to pose questions. The answers should be concise and accurate. If you find the provided text too brief or lacking sufficient content to generate questions, no response is needed.

```

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

## Configuration Options

- `llm_client`: Choose between 'azure' or 'openai'
- `llm_model`: Specify custom model (defaults available for each client)
- `chunk_size`: Control the size of text chunks (default: 4096)
- `entries_per_chunk`: Number of QA pairs to generate per chunk (default: 3)
- `language`: Choose between 'zhtw' (Traditional Chinese) or 'en' (English)

## Best Practices

1. Start with a small test file before processing large directories
2. Monitor the generated output quality
3. Adjust chunk size based on your content
4. Use appropriate language setting for your source material
5. Consider using custom prompts for specific use cases

## Contact Me

Have questions or suggestions? Feel free to reach out!

Email: [qqandy0120@gmail.com](mailto:qqandy0120@gmail.com)