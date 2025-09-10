import torch
import asyncio
from openai import AsyncOpenAI, APIConnectionError, Timeout

class Reranker:
    def __init__(self, emb_client: AsyncOpenAI, model_name="Qwen/Qwen3-Embedding-0.6B"):
        self.emb_client = emb_client
        self.model_name = model_name

    async def embed(self, input_texts: list, max_retries: int = 3) -> torch.Tensor:
        for attempt in range(max_retries):
            try:
                results = await self.emb_client.embeddings.create(input=input_texts, model=self.model_name)
                embeddings = torch.tensor([d.embedding for d in results.data])
                return embeddings
            except (APIConnectionError, Timeout, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 10))  # Exponential backoff
                else:
                    return torch.zeros((len(input_texts), 1024))  # Assuming embedding size is 1024

    async def compute_similarity(self, queries: list, documents: list) -> torch.Tensor:
        input_texts = queries + documents
        embeddings = await self.embed(input_texts)
        query_embeddings = embeddings[:len(queries)]
        document_embeddings = embeddings[len(queries):]
        scores = query_embeddings @ document_embeddings.T
        return scores