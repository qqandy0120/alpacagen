## AlpacaGen
from typing import Dict
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    source: str
    idx: str  ## format: 01/17

@dataclass
class QAPair:
    instruction: str
    input: str
    output: str
    source: Chunk

    def to_dict(self) -> Dict[str, str]:
        return {
            'instruction': str(self.instruction),
            'input': str(self.input),
            'output': str(self.output),
            'source': repr(Chunk)
        }