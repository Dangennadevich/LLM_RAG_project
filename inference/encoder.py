import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)

    @torch.no_grad()
    def encode(self, user_query) -> np.array:

        embeddings = self.model.encode(user_query, max_length=1024)["dense_vecs"]
        return embeddings
