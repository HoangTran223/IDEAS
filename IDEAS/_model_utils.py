import torch
import numpy as np
from sentence_transformers import SentenceTransformer

import logging

class DocEmbedModel:
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = False,
        device: str = "cpu",
        verbose: bool = False
    ):
        self.verbose = verbose
        self.normalize_embeddings = normalize_embeddings

        if isinstance(model, str):
            self.model = SentenceTransformer(model, device=device)
        else:
            self.model = model

    def encode(self, docs: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            docs,
            show_progress_bar=self.verbose,
            normalize_embeddings=self.normalize_embeddings
        )
        return embeddings


def pairwise_euclidean_distance(x, y):
    cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
    return cost
