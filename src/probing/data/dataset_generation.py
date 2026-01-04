import csv
import json
import os
import asyncio
from pathlib import Path
from pydantic import BaseModel
import httpx
from src.prompts.system import PROBING_DATASET_PROMPT
from src.probing.data.probing_dataset import HuggingFaceDatasetLoader

from dotenv import load_dotenv
load_dotenv()

class LLMResponse(BaseModel):
    responses: list[str]


class GeminiBatchProcessor:
    def __init__(self, api_key: str = None, batch_size: int = 25, output_file: str = "output.csv", delay: float = 2.0):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.batch_size = batch_size
        self.output_file = Path(output_file)
        self.delay = delay  # Delay between batches in seconds
        
        if not self.output_file.exists():
            self.output_file.write_text("input,output,batch_id\n")
    
    async def _call_api(self, texts: list[str], system_prompt: str) -> list[str]:
        prompt = "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": "google/gemini-3-flash-preview",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "response",
                                    "strict": True,
                                    "schema": LLMResponse.model_json_schema()
                                }
                            }
                        }
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    return json.loads(content)["responses"]
                except Exception as e:
                    if attempt == 2:
                        return [f"ERROR: {e}"] * len(texts)
                    # Exponential backoff: 2s, 4s, 8s
                    await asyncio.sleep(2 ** (attempt + 1))
    
    def _save_batch(self, texts: list[str], responses: list[str], batch_id: int):
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for text, resp in zip(texts, responses):
                writer.writerow([text, resp, batch_id])
    
    async def process_async(self, texts: list[str], system_prompt: str):
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            print(f"Processing batch {batch_num}...")
            responses = await self._call_api(batch, system_prompt)
            self._save_batch(batch, responses, i // self.batch_size)
            
            # Add delay between batches to avoid rate limits
            if i + self.batch_size < len(texts):
                print(f"Waiting {self.delay}s before next batch...")
                await asyncio.sleep(self.delay)
    
    def process(self, texts: list[str], system_prompt: str):
        asyncio.run(self.process_async(texts, system_prompt))

if __name__ == "__main__":
    loader = HuggingFaceDatasetLoader()
    dataset = loader.load_dataset("Anthropic/persuasion_train")['claim']
    print(f"Total examples to process: {len(dataset)}")
    corpus = list(set(dataset))
    print(f"Total unique examples to process: {len(corpus)}")
    print(dataset[:100])

    # processor = GeminiBatchProcessor(batch_size=25, output_file="output.csv")
    # processor.process(texts, PROBING_DATASET_PROMPT)