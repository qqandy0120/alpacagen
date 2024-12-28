## AlpacaGen
from typing import Dict
from dataclasses import dataclass

@dataclass
class QAPair:
    instruction: str
    input: str
    output: str
    
    @property
    def text(self) -> str:
        return (
            f"以下是一個描述任務，配對提供進一步上下文的輸入。請適當地完成請求。"
            f"### Instruction: {self.instruction}\n"
            f"### Input: {self.input}\n"
            f"### Response: {self.output}"
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'instruction': str(self.instruction),
            'input': str(self.input),
            'output': str(self.output),
            'text': str(self.text)
        }

@dataclass
class Chunk:
    content: str
    source: str
    idx: str  ## format: 01/17
    # TODO: function or property for Chunk