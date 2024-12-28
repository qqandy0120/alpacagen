import json
import logging
from typing import Optional
from openai import AsyncOpenAI
from ..models.qa_pair import QAPair

logger = logging.getLogger(__name__)

class QAGenerator:
    def __init__(self, client: AsyncOpenAI, llm_model: str, prompt_template: str):
        self.client = client
        self.llm_model = llm_model
        self.prompt_template = prompt_template
    
    async def generate(self, text: str) -> Optional[QAPair]:
        try:
            response = await self.client.completions.create(
                model=self.llm_model,
                prompt=self.prompt_template.format(text=text),
                max_tokens=1024,
            )
            
            response_text = response.choices[0].text
            if not self._is_valid_json(response_text):
                logger.info("Unable to receive the expected response in JSON format.")
                return None
            
            data = json.loads(response_text)
            return QAPair(
                instruction=data['instruction'],
                input=data['input'],
                output=data['output']
            )
            
        except Exception as e:
            logger.error(f"Error during generating entry: {str(e)}")
            return None
    
    @staticmethod
    def _is_valid_json(response: str) -> bool:
        try:
            data = json.loads(response)
            return all(key in data for key in ['instruction', 'input', 'output'])
        except:
            return False