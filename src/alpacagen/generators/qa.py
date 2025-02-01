import json
import logging
from typing import Optional, List
from openai import AsyncOpenAI
from ..models.qa_pair import QAPair, Chunk
from ..generators.client import BaseLLMClient
logger = logging.getLogger(__name__)

class QAGenerator:
    def __init__(self, client: BaseLLMClient, llm_model: str, prompt_template: str) -> List[QAPair]:
        self.client = client
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        self.max_retry_time = 3
    '''
    Chunk -> List[QAPair]
    '''
    async def generate(self, chunk: Chunk, entries_per_chunk, retry_time=0) -> Optional[List[QAPair]]:
        if retry_time >= self.max_retry_time:
            return None
        
        try:
            response_text = await self.client.get_response(
                prompt=self.prompt_template.format(
                    text=chunk.content, 
                    entries_per_chunk=entries_per_chunk
                ),
                max_tokens=1024,
            )
            # print(response_text)
            response_json_format = self.parsing_response(response_text)
            # Add validation check
            if not response_json_format:  # If empty or None
                logger.info("Unable to receive the expected response in JSON format.")
                return await self.generate(chunk, entries_per_chunk, retry_time + 1)

            entries = []
            for response in response_json_format:
                try:
                    response_entry = json.loads(response)
                    logger.info(f"Response: {response}")
                    entries.append(
                        QAPair(
                            instruction=response_entry['instruction'],
                            input=response_entry['input'],
                            output=response_entry['output'],
                            source=chunk,
                        )
                    )
                except json.JSONDecodeError as e:
                    logger.info(f"JSON decode error: {e}")
                    continue
                except KeyError as e:
                    logger.info(f"Missing required key: {e}")
                    continue

            if not entries:  # If no valid entries were created
                return await self.generate(chunk, entries_per_chunk, retry_time + 1)
                
            return entries
                
        except Exception as e:
            logger.error(f"Error during generating entry: {str(e)}")
            return await self.generate(chunk, entries_per_chunk, retry_time + 1)
    
    @staticmethod
    def parsing_response(response: str):
        json_objects = [obj.strip() for obj in response.split('}\n') if obj.strip()]
        json_objects = [obj + '}' for obj in json_objects[:-1]] + [json_objects[-1]]
        return json_objects