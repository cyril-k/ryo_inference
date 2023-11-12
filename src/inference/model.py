import os
from typing import (
    Dict,
)

import sentence_transformers

MODEL_NAME = os.getenv("MODEL_NAME") #replace with "model_checkpoint" if you didnt download it
# MODEL_NAME = "thenlper/gte-base" #replace with "model_checkpoint" if you didnt download it

class Model:
    def __init__(self, data_dir: str, config: Dict, **kwargs) -> None:
        self._data_dir = data_dir
        self._config = config
        self.device = "cpu" #if torch.cuda.is_available() else "cpu"
        print("THE DEVICE INFERENCE IS RUNNING ON IS: ", self.device)
        # self.tokenizer = None
        # self.pipeline = None
        # self.stopping_criteria = None

    def load(self):
        # Load model here and assign to self._model.
        self._model = sentence_transformers.SentenceTransformer(
            MODEL_NAME,
            # cache_folder="~/.cache/torch/sentence_transformers",
            # **self.model_kwargs
            device=self.device
        )

    def predict(self, request:Dict):
        # Run model inference here
        texts = request.pop("texts")
        if isinstance(texts, list):
            print(f"Computing embeddings for {len(texts)} chunks.")
        else:
            print(f"computing embeddings for: {texts}")

        embeddings = self._model.encode(
            texts, 
            **request
        )

        return embeddings.tolist()

