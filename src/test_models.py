
import unittest
import torch
from attention_retriever import AttentionBasedRetriever
from baselines import BaselineRetrievers

class TestAttentionBasedRetriever(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 768
        self.batch_size = 4
        self.query_embeddings = torch.randn(self.batch_size, self.embedding_dim)
        self.document_embeddings = torch.randn(self.batch_size, self.embedding_dim)

    def test_equation1_output_shape(self):
        model = AttentionBasedRetriever(self.embedding_dim, extended_version=False)
        output = model(self.query_embeddings.to(model.device), self.document_embeddings.to(model.device))
        self.assertEqual(output.shape, (self.batch_size,))

    def test_equation2_output_shape(self):
        model = AttentionBasedRetriever(self.embedding_dim, extended_version=True)
        output = model(self.query_embeddings.to(model.device), self.document_embeddings.to(model.device))
        self.assertEqual(output.shape, (self.batch_size,))

    def test_equation1_output_range(self):
        model = AttentionBasedRetriever(self.embedding_dim, extended_version=False)
        output = model(self.query_embeddings.to(model.device), self.document_embeddings.to(model.device))
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))

    def test_equation2_output_range(self):
        model = AttentionBasedRetriever(self.embedding_dim, extended_version=True)
        output = model(self.query_embeddings.to(model.device), self.document_embeddings.to(model.device))
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))

class TestBaselineRetrievers(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 768
        self.batch_size = 4
        self.query_embeddings = torch.randn(self.batch_size, self.embedding_dim)
        self.document_embeddings = torch.randn(self.batch_size, self.embedding_dim)

    def test_cosine_similarity_output_shape(self):
        scores = BaselineRetrievers.cosine_similarity_retriever(self.query_embeddings, self.document_embeddings)
        self.assertEqual(scores.shape, (self.batch_size,))

    def test_dot_product_output_shape(self):
        scores = BaselineRetrievers.dot_product_retriever(self.query_embeddings, self.document_embeddings)
        self.assertEqual(scores.shape, (self.batch_size,))

    def test_cosine_similarity_range(self):
        scores = BaselineRetrievers.cosine_similarity_retriever(self.query_embeddings, self.document_embeddings)
        self.assertTrue(all((score >= -1.0 and score <= 1.0) for score in scores))

    def test_bm25_placeholder(self):
        # This is a placeholder test for BM25 as it's a conceptual implementation
        query_text = "test query"
        document_texts = ["doc1", "doc2"]
        scores = BaselineRetrievers.bm25_retriever(query_text, document_texts)
        self.assertEqual(len(scores), len(document_texts))

if __name__ == '__main__':
    unittest.main()


