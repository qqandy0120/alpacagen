# tests/test_alpacagen.py
import pytest
from pathlib import Path
from alpacagen import (
    AlpacaGen,
    QAPair,
    Chunk,
    RecursiveChunkStrategy,
    MarkItDownConverter
)

# Test data
SAMPLE_TEXT = """# Test Document
This is a test document with multiple paragraphs.

## Section 1
This is the first section of the document.
It contains some text that should be processed.

## Section 2
This is the second section with more content.
"""

@pytest.fixture
def sample_text_file(tmp_path):
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text(SAMPLE_TEXT)
    return file_path

@pytest.fixture
def mock_openai_client(mocker):
    mock_client = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.choices = [
        mocker.Mock(
            text='{"instruction": "Summarize the content", "input": "Given text", "output": "Summary"}'
        )
    ]
    mock_client.completions.create.return_value = mock_response
    return mock_client

class TestQAPair:
    def test_qa_pair_creation(self):
        qa_pair = QAPair(
            instruction="Test instruction",
            input="Test input",
            output="Test output"
        )
        assert qa_pair.instruction == "Test instruction"
        assert qa_pair.input == "Test input"
        assert qa_pair.output == "Test output"

    def test_qa_pair_to_dict(self):
        qa_pair = QAPair(
            instruction="Test instruction",
            input="Test input",
            output="Test output"
        )
        result = qa_pair.to_dict()
        assert isinstance(result, dict)
        assert result["instruction"] == "Test instruction"
        assert result["input"] == "Test input"
        assert result["output"] == "Test output"
        assert "text" in result

class TestChunkStrategy:
    def test_recursive_chunk_strategy(self):
        strategy = RecursiveChunkStrategy(chunk_size=100, chunk_overlap=20)
        chunks = strategy.split("test.txt", SAMPLE_TEXT)
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.content) <= 100 for chunk in chunks)

class TestMarkItDownConverter:
    def test_converter_with_text_file(self, sample_text_file):
        converter = MarkItDownConverter()
        result = converter.convert(sample_text_file)
        
        assert isinstance(result, str)
        assert "Test Document" in result
        assert "Section 1" in result
        assert "Section 2" in result

class TestAlpacaGen:
    @pytest.mark.asyncio
    async def test_generate_single_file(self, sample_text_file, mock_openai_client, tmp_path):
        output_file = tmp_path / "output.jsonl"
        
        ag = AlpacaGen(
            llm_client='azure',
            api_key='test-key',
            base_url='test-url'
        )
        
        chunks, dataset = ag.generate(
            input_path=sample_text_file,
            output_path=output_file,
            entries_per_chunk=1
        )
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert isinstance(dataset, list)
        assert all(isinstance(entry, QAPair) for entry in dataset)
        assert output_file.exists()

    def test_invalid_language(self):
        ag = AlpacaGen(
            llm_client='azure',
            api_key='test-key',
            base_url='test-url'
        )
        
        with pytest.raises(AssertionError):
            ag.generate(
                input_path="test.txt",
                output_path="output.jsonl",
                language='invalid'
            )

if __name__ == '__main__':
    pytest.main([__file__])