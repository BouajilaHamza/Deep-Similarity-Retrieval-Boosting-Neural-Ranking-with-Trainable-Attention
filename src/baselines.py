

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BaselineRetrievers:
    @staticmethod
    def cosine_similarity_retriever(query_embedding, document_embedding):
        # Ensure inputs are numpy arrays for sklearn or convert to torch for direct computation
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        if isinstance(document_embedding, torch.Tensor):
            document_embedding = document_embedding.cpu().numpy()

        # Reshape for single sample if necessary
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if document_embedding.ndim == 1:
            document_embedding = document_embedding.reshape(1, -1)

        # Compute cosine similarity. If batched, iterate or use broadcasting if possible.
        # For simplicity, assuming batch_size for both is the same and we want pairwise similarity
        # If query_embedding is (batch_size, dim) and document_embedding is (batch_size, dim)
        # we want (batch_size,) output where each element is sim(q_i, d_i)
        
        # Using torch for batched cosine similarity for efficiency
        query_embedding_t = torch.from_numpy(query_embedding)
        document_embedding_t = torch.from_numpy(document_embedding)

        # Normalize embeddings
        query_norm = query_embedding_t.norm(p=2, dim=1, keepdim=True)
        document_norm = document_embedding_t.norm(p=2, dim=1, keepdim=True)
        
        query_embedding_normalized = query_embedding_t / query_norm
        document_embedding_normalized = document_embedding_t / document_norm

        # Compute dot product of normalized embeddings
        # Element-wise multiplication and sum along the dimension
        similarity_scores = torch.sum(query_embedding_normalized * document_embedding_normalized, dim=1)
        return similarity_scores.numpy()

    @staticmethod
    def dot_product_retriever(query_embedding, document_embedding):
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        if isinstance(document_embedding, torch.Tensor):
            document_embedding = document_embedding.cpu().numpy()

        # Element-wise multiplication and sum along the dimension
        similarity_scores = np.sum(query_embedding * document_embedding, axis=1)
        return similarity_scores

    # BM25 will require actual text, not just embeddings. This will be handled in the experiment setup.
    # For now, a placeholder or a simplified version for conceptual completeness.
    @staticmethod
    def bm25_retriever(query_text, document_texts, corpus=None):
        # This is a simplified conceptual placeholder. A real BM25 implementation
        # would involve a library like `rank_bm25` and a pre-built corpus.
        # For now, we'll just return dummy scores.
        print("BM25 retriever requires text data and a corpus. This is a placeholder.")
        return np.random.rand(len(document_texts))

if __name__ == '__main__':
    embedding_dim = 768
    batch_size = 4

    query_embeddings = torch.randn(batch_size, embedding_dim)
    document_embeddings = torch.randn(batch_size, embedding_dim)

    # Test Cosine Similarity
    cos_sim_scores = BaselineRetrievers.cosine_similarity_retriever(query_embeddings, document_embeddings)
    print(f"Cosine Similarity Scores: {cos_sim_scores.shape} - {cos_sim_scores}")

    # Test Dot Product
    dot_prod_scores = BaselineRetrievers.dot_product_retriever(query_embeddings, document_embeddings)
    print(f"Dot Product Scores: {dot_prod_scores.shape} - {dot_prod_scores}")

    # Test BM25 (conceptual)
    query_text = "example query"
    document_texts = ["example document one", "another example document"]
    bm25_scores = BaselineRetrievers.bm25_retriever(query_text, document_texts)
    print(f"BM25 Scores: {bm25_scores.shape} - {bm25_scores}")


