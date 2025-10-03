import torch
from openai import AsyncOpenAI

class Reranker:
    def __init__(self, emb_client: AsyncOpenAI, model_name="Qwen/Qwen3-Embedding-0.6B"):
        """
        Initializes the Reranker with an async embedding client.

        Args:
            emb_client: An async embedding client instance (e.g., AsyncOpenAI client).
            model_name: Name of the embedding model to use.
        """
        self.emb_client = emb_client
        self.model_name = model_name

    async def embed(self, input_texts: list) -> torch.Tensor:
        """
        Embeds the input texts using the async embedding client.

        Args:
            input_texts (list): A list of strings to embed.

        Returns:
            torch.Tensor: A tensor containing the embeddings for the input texts.
        """
        # Use the async embedding client to generate embeddings
        results = await self.emb_client.embeddings.create(input=input_texts, model=self.model_name)
        embeddings = torch.tensor([d.embedding for d in results.data])
        return embeddings

    async def compute_similarity(self, queries: list, documents: list) -> torch.Tensor:
        """
        Computes similarity scores between queries and documents.

        Args:
            queries (list): A list of query strings.
            documents (list): A list of document strings.

        Returns:
            torch.Tensor: A tensor containing similarity scores between queries and documents.
        """
        input_texts = queries + documents
        embeddings = await self.embed(input_texts)
        query_embeddings = embeddings[:len(queries)]
        document_embeddings = embeddings[len(queries):]
        scores = query_embeddings @ document_embeddings.T
        return scores