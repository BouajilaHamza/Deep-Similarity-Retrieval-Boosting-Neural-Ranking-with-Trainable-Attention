
import torch
import torch.nn as nn
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer

from attention_retriever import AttentionBasedRetriever
from baselines import BaselineRetrievers

import logging
import os
import numpy as np
from tqdm import tqdm

#### Just some code to print debug information to stdout
logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

class CustomDenseEncoder(nn.Module):
    def __init__(self, sentence_transformer_model, custom_similarity_model=None):
        super(CustomDenseEncoder, self).__init__()
        self.sentence_transformer_model = sentence_transformer_model
        self.custom_similarity_model = custom_similarity_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentence_transformer_model.to(self.device)
        if self.custom_similarity_model:
            self.custom_similarity_model.to(self.device)

    def encode_queries(self, queries, batch_size=16):
        # queries is a list of strings
        return self.sentence_transformer_model.encode(queries, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)

    def encode_corpus(self, corpus, batch_size=16):
        # corpus is a dictionary {doc_id: {"text": "doc text", "title": "doc title"}}
        corpus_sentences = [corpus[cid]["title"] + " " + corpus[cid]["text"] for cid in corpus]
        return self.sentence_transformer_model.encode(corpus_sentences, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)

    def compute_custom_similarity(self, query_embedding, document_embedding):
        # query_embedding and document_embedding are numpy arrays
        query_embedding_t = torch.from_numpy(query_embedding).float().to(self.device)
        document_embedding_t = torch.from_numpy(document_embedding).float().to(self.device)
        
        # Ensure batch dimension for single samples
        if query_embedding_t.ndim == 1:
            query_embedding_t = query_embedding_t.unsqueeze(0)
        if document_embedding_t.ndim == 1:
            document_embedding_t = document_embedding_t.unsqueeze(0)

        with torch.no_grad():
            score = self.custom_similarity_model(query_embedding_t, document_embedding_t).cpu().numpy()
        return score.item() # Return scalar score

def run_experiment(dataset_name, model_type, custom_model_name=None, output_path="./results"):
    #### Download dataset and load corpus, queries, and relevant documents
    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_name).load(split="test")

    # Load a pre-trained Sentence-BERT model for initial embeddings
    # This model will be used to generate the base embeddings for all retrievers.
    logging.info("Loading Sentence-BERT model...")
    sentence_transformer_model = SentenceTransformer("msmarco-distilbert-base-v4")
    embedding_dim = sentence_transformer_model.get_sentence_embedding_dimension()

    if model_type == "attention_retriever":
        if custom_model_name == "attention_retriever_eq1":
            custom_model = AttentionBasedRetriever(embedding_dim=embedding_dim, extended_version=False)
        elif custom_model_name == "attention_retriever_eq2":
            custom_model = AttentionBasedRetriever(embedding_dim=embedding_dim, extended_version=True)
        else:
            raise ValueError("Invalid custom_model_name for attention_retriever type.")
        
        # For attention-based retriever, we need to implement a custom search logic
        # that uses the encoded embeddings and then applies our attention model.
        # BEIR\s DRES is not directly compatible with a custom similarity function.
        # So, we will manually compute scores and then format for BEIR evaluation.
        logging.info(f"Encoding queries and corpus for {custom_model_name}...")
        query_encoder = CustomDenseEncoder(sentence_transformer_model)
        corpus_encoder = CustomDenseEncoder(sentence_transformer_model)

        query_embeddings = query_encoder.encode_queries(list(queries.values()))
        corpus_embeddings = corpus_encoder.encode_corpus(corpus)

        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())

        results = {}
        logging.info(f"Running {custom_model_name} on {dataset_name}...")

        # Convert corpus_embeddings to a tensor for efficient batch processing
        corpus_embeddings_tensor = torch.from_numpy(corpus_embeddings).float().to(custom_model.device)

        for i, qid in tqdm(enumerate(query_ids), total=len(query_ids), desc="Processing Queries"):
            query_vec = torch.from_numpy(query_embeddings[i]).float().unsqueeze(0).to(custom_model.device) # (1, embedding_dim)
            
            # Expand query_vec to match the batch size of corpus_embeddings_tensor
            # (num_corpus_docs, embedding_dim)
            expanded_query_vec = query_vec.expand(corpus_embeddings_tensor.shape[0], -1)

            with torch.no_grad():
                # Compute similarity for all corpus documents in one go
                scores = custom_model(expanded_query_vec, corpus_embeddings_tensor).cpu().numpy()
            
            # Convert scores to native Python floats
            doc_scores = {cid: float(score) for cid, score in zip(corpus_ids, scores)}
            
            # Sort documents by score and take top_k
            sorted_doc_scores = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
            results[qid] = {cid: score for cid, score in sorted_doc_scores[:100]} # Top 100 for evaluation

    elif model_type == "baseline":
        # For baselines, we can use DRES directly with the encoded embeddings
        logging.info(f"Encoding queries and corpus for {custom_model_name}...")
        model = CustomDenseEncoder(sentence_transformer_model)
        
        # BEIR\s DRES uses dot product by default. For cosine similarity, embeddings need to be normalized.
        if custom_model_name == "cosine_similarity":
            # Normalize embeddings for cosine similarity
            class CosineSimilarityEncoder(CustomDenseEncoder):
                def encode_queries(self, queries, batch_size=16):
                    embeddings = super().encode_queries(queries, batch_size)
                    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                def encode_corpus(self, corpus, batch_size=16):
                    embeddings = super().encode_corpus(corpus, batch_size)
                    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            model = CosineSimilarityEncoder(sentence_transformer_model)
            
        elif custom_model_name == "dot_product":
            pass # No special normalization needed for dot product
        elif custom_model_name == "bm25":
            # BM25 is a sparse retriever, handled differently by BEIR
            # We will use a separate BM25 implementation if needed, not DRES.
            # For now, we skip BM25 in this DRES-based flow.
            logging.info("BM25 is not supported in this DRES-based flow. Skipping.")
            return
        else:
            raise ValueError("Invalid custom_model_name for baseline type.")

        logging.info(f"Running {custom_model_name} on {dataset_name}...")
        retriever = DRES(model, batch_size=16)
        results = retriever.retrieve(corpus, queries)

        # Ensure scores are native Python floats for DRES results as well
        for qid in results:
            for cid in results[qid]:
                results[qid][cid] = float(results[qid][cid])

    else:
        raise ValueError("Unknown model_type.")

    #### Evaluate your retrieval using BEIR\s evaluation framework
    logging.info("Evaluating retrieval results...")
    retriever_evaluator = EvaluateRetrieval()
    
    # BEIR expects results in a specific format: {query_id: {doc_id: score, ...}, ...}
    # Our `results` dictionary is already in this format.

    # BEIR metrics: nDCG@k, MAP@k, Recall@k, P@k
    ndcg, map_score, recall, p = retriever_evaluator.evaluate(qrels, results, retriever_evaluator.k_values)

    logging.info(f"Results for {custom_model_name} on {dataset_name}:")
    for k in retriever_evaluator.k_values:
        logging.info(f"nDCG@{k}: {ndcg[f'nDCG@{k}']:.4f}")
        logging.info(f"MAP@{k}: {map_score[f'MAP@{k}']:.4f}")
        logging.info(f"Recall@{k}: {recall[f'Recall@{k}']:.4f}")
        logging.info(f"P@{k}: {p[f'P@{k}']:.4f}")

    # Save results
    output_dir = os.path.join(output_path, dataset_name, custom_model_name)
    os.makedirs(output_dir, exist_ok=True)
    retriever_evaluator.save(results, output_dir)
    logging.info(f"Results saved to {output_dir}")

if __name__ == '__main__':
    # Example usage:
    dataset_to_test = "nfcorpus"
    
    # Download dataset if not already present
    data_path = os.path.join(os.getcwd(), dataset_to_test)
    if not os.path.exists(data_path):
        logging.info(f"Downloading {dataset_to_test} dataset...")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_to_test)
        util.download_and_unzip(url, os.getcwd())

    # Run experiments for each model
    models_to_benchmark = [
        ("attention_retriever", "attention_retriever_eq1"),
        ("attention_retriever", "attention_retriever_eq2"),
        ("baseline", "cosine_similarity"),
        ("baseline", "dot_product")
    ]

    for model_type, model_name in models_to_benchmark:
        run_experiment(dataset_to_test, model_type, model_name)



