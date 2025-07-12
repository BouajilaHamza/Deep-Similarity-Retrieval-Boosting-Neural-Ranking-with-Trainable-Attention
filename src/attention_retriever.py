
import torch
import torch.nn as nn

class AttentionBasedRetriever(nn.Module):
    def __init__(self, embedding_dim, extended_version=False):
        super(AttentionBasedRetriever, self).__init__()
        self.embedding_dim = embedding_dim
        self.extended_version = extended_version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not extended_version:
            # Equation 1: sim(q,d) = sigma(q^T M d + b)
            self.M = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
            self.b = nn.Parameter(torch.randn(1))
        else:
            # Equation 2: sim(q,d) = sigma(W2 * phi(W1 * [q;d;q*d]) + b)
            # [q;d;q*d] will have dimension 3 * embedding_dim
            self.W1 = nn.Linear(3 * embedding_dim, embedding_dim) # Example, can be different
            self.W2 = nn.Linear(embedding_dim, 1)
            self.phi = nn.Tanh() # Non-linear activation function
        
        self.to(self.device)

    def forward(self, query_embedding, document_embedding):
        if not self.extended_version:
            # Equation 1
            transformed_d = torch.matmul(self.M, document_embedding.transpose(0, 1)).transpose(0, 1) # (batch_size, embedding_dim)
            similarity_score = torch.sum(query_embedding * transformed_d, dim=1) + self.b.squeeze()
            return torch.sigmoid(similarity_score)
        else:
            # Equation 2
            concatenated_embedding = torch.cat([query_embedding, document_embedding, query_embedding * document_embedding], dim=1)
            hidden_state = self.phi(self.W1(concatenated_embedding))
            similarity_score = self.W2(hidden_state).squeeze()
            return torch.sigmoid(similarity_score)

if __name__ == '__main__':
    # Test the implementation
    embedding_dim = 768 # Common embedding dimension for models like BERT
    batch_size = 4

    query_embeddings = torch.randn(batch_size, embedding_dim)
    document_embeddings = torch.randn(batch_size, embedding_dim)

    # Test Equation 1
    model_eq1 = AttentionBasedRetriever(embedding_dim, extended_version=False)
    output_eq1 = model_eq1(query_embeddings.to(model_eq1.device), document_embeddings.to(model_eq1.device))
    print(f"Output (Equation 1): {output_eq1.shape} - {output_eq1}")

    # Test Equation 2
    model_eq2 = AttentionBasedRetriever(embedding_dim, extended_version=True)
    output_eq2 = model_eq2(query_embeddings.to(model_eq2.device), document_embeddings.to(model_eq2.device))
    print(f"Output (Equation 2): {output_eq2.shape} - {output_eq2}")

    # Test with single example
    query_single = torch.randn(1, embedding_dim)
    document_single = torch.randn(1, embedding_dim)
    output_single_eq1 = model_eq1(query_single.to(model_eq1.device), document_single.to(model_eq1.device))
    print(f"Output (Equation 1, single): {output_single_eq1.shape} - {output_single_eq1}")

    output_single_eq2 = model_eq2(query_single.to(model_eq2.device), document_single.to(model_eq2.device))
    print(f"Output (Equation 2, single): {output_single_eq2.shape} - {output_single_eq2}")


