## AlpacaGen
import logging
from typing import Union
from pathlib import Path
from abc import ABC, abstractmethod
from markitdown import MarkItDown

class TextConverter(ABC):
    @abstractmethod
    def convert(self, input_path: Union[str, Path]) -> str:
        pass

class MarkItDownConverter(TextConverter):
    """ use MakeItDown as a converter """
    def convert(self, input_path: Union[str, Path]) -> str:
        """ input path should be a file, not a directory """
        md = MarkItDown()

        if isinstance(input_path, Path):
            input_path = str(input_path)

        try:
            extract_text = md.convert(input_path).text_content
            if not extract_text:
                logging.warning(f"{input_path} is empty after conversion.")
            
            return extract_text
        
        except Exception as e:
            logging.error(f"Error extracting file{input_path}: {str(e)}")
            raise